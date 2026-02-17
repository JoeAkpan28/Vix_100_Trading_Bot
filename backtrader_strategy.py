"""
BACKTRADER MARKOV CHAIN TRADING STRATEGY
=========================================
Professional backtrader implementation of the Markov predictor.
This converts the walk-forward analysis into a proper backtrader strategy
with zero data leakage and professional position management.

Key Features:
- Integrates MarkovLivePredictor with backtrader
- Maintains rolling state queue (last 3 states)
- Proper order management (position sizing, stop-loss, take-profit)
- Detailed logging and performance metrics
- Compatible with live trading via backtrader's live capabilities
"""

import backtrader as bt
import pandas as pd
import numpy as np
from markov_predictor import MarkovLivePredictor
from sl_logic import StopLossCalculator


class MarkovStrategy(bt.Strategy):
    """
    Markov Chain based trading strategy for backtrader.
    
    Uses a 3rd-order Markov model to predict UP/DOWN movements
    based on historical candle patterns with asymmetrical confidence thresholds.
    
    Parameters:
    - up_threshold: Probability threshold for UP signal (default: 0.50)
    - conf_up: Confidence offset for UP signals (default: 0.15, generates UP if up_probability > 0.65)
    - conf_down: Confidence offset for DOWN signals (default: 0.40, generates DOWN if up_probability < 0.10)
    - position_size: % of equity to risk per trade (default: 0.01)
    - stop_loss_pct: Stop loss distance in % (default: 1.0)
    - take_profit_pct: Take profit distance in % (default: 2.0)
    - matrix_path: Path to markov_brain.json (default: 'markov_brain.json')
    - window_size: Number of candles for state calculation (default: 100)
    """
    
    params = (
        ('up_threshold', 0.50),
        ('conf_up', 0.15),      # UP signal threshold offset: UP if up_prob > 0.50 + 0.15 = 0.65
        ('conf_down', 0.40),    # DOWN signal threshold offset: DOWN if up_prob < 0.50 - 0.40 = 0.10
        ('position_size', 0.01),  # 1% of equity per trade
        ('stop_loss_pct', 1.0),
        ('take_profit_pct', 2.0),
        ('matrix_path', 'markov_brain.json'),
        ('window_size', 100),
    )
    
    def __init__(self):
        """Initialize the strategy."""
        # Initialize Markov predictor with asymmetrical thresholds
        self.predictor = MarkovLivePredictor(
            matrix_path=self.params.matrix_path,
            up_threshold=self.params.up_threshold,
            conf_up=self.params.conf_up,
            conf_down=self.params.conf_down
        )
        
        # Initialize SL calculator
        self.sl_calculator = StopLossCalculator(
            window_size=self.params.window_size,
            pip_size=1.0,
            percentile=0.95
        )
        
        # State management
        self.state_queue = []  # Last 3 states
        self.bar_count = 0
        self.last_prediction = None
        self.entry_price = None
        self.entry_signal = None
        self.sl_price = None
        self.tp_price = None
        
        # Order tracking
        self.pending_order = None
        self.order_signal = None
        self.order_entry_price = None
        self.order_sl_buffer = None
        self.order_state_0 = None
        self.order_state_1 = None
        self.order_state_2 = None
        
        # Logging
        self.trade_log = []
        self.sl_buffer = None
        
        print("\n" + "="*70)
        print("MARKOV STRATEGY INITIALIZED")
        print("="*70)
        print(f"   UP Threshold: {self.params.up_threshold}")
        print(f"   UP Signal (conf_up): up_probability > {self.params.up_threshold + self.params.conf_up:.2f}")
        print(f"   DOWN Signal (conf_down): up_probability < {self.params.up_threshold - self.params.conf_down:.2f}")
        print(f"   Position Size: {self.params.position_size*100:.1f}% of equity")
        print(f"   Stop Loss: {self.params.stop_loss_pct}%")
        print(f"   Take Profit: {self.params.take_profit_pct}%")
        print(f"   Window Size: {self.params.window_size}")
        print(f"   SL Calculator: Dynamic (based on entry wicks)")
        print("="*70)
    
    def next(self):
        """Called on each bar (candle) to process new data."""
        self.bar_count += 1
        
        # Need at least window_size bars to make a prediction
        if self.bar_count < self.params.window_size:
            return
        
        # Build the window for state calculation
        # We need the last N candles (including current)
        window_size = self.params.window_size
        
        if len(self) < window_size:
            return
        
        try:
            # Create a DataFrame from the last N bars
            # This ensures zero data leakage - we only use data available NOW
            bars_data = {
                'Open': [self.data.open[-i] for i in range(window_size-1, -1, -1)],
                'High': [self.data.high[-i] for i in range(window_size-1, -1, -1)],
                'Low': [self.data.low[-i] for i in range(window_size-1, -1, -1)],
                'Close': [self.data.close[-i] for i in range(window_size-1, -1, -1)],
            }
            
            window_df = pd.DataFrame(bars_data)
            
            # Calculate SL buffer for current bar
            window_df = self.sl_calculator.calculate_entry_wick(
                window_df, open_col='Open', close_col='Close', 
                high_col='High', low_col='Low'
            )
            window_df = self.sl_calculator.calculate_sl_buffer(
                window_df, entry_wick_col='entry_wick'
            )
            
            # Get SL buffer for current bar (last row)
            self.sl_buffer = window_df['sl_buffer'].iloc[-1]
            
            # Create lowercase version for predictor (expects lowercase column names)
            predictor_df = window_df[['Open', 'High', 'Low', 'Close']].copy()
            predictor_df.columns = ['open', 'high', 'low', 'close']
            
            # Get the current state
            current_state = self.predictor.get_current_state(predictor_df)
            self.state_queue.append(current_state)
            
            # Keep only last 3 states
            if len(self.state_queue) > 3:
                self.state_queue.pop(0)
            
            # Make prediction if we have 3 states
            if len(self.state_queue) == 3:
                sequence = tuple(self.state_queue)
                prediction = self.predictor.predict(sequence, verbose=False)
                self.last_prediction = prediction
                
                # ONLY log bars where we have a valid signal that executes a trade
                # Skip logging UNKNOWN signals to avoid fake entry/exit tracking
                log_entry = {
                    'bar': len(self),
                    'datetime': self.data.datetime.date(0),
                    'state_0': int(sequence[0]),
                    'state_1': int(sequence[1]),
                    'state_2': int(sequence[2]),
                    'up_prob': prediction['up_probability'],
                    'confidence': prediction['confidence'],
                    'signal': prediction['signal'],
                    'price': self.data.close[0],
                    'sl_buffer': self.sl_buffer,
                    'position_open': 1 if self.position else 0,
                    'filter_reason': prediction.get('filter_reason', 'none'),
                    'threshold_used': prediction.get('threshold_used', 'none')
                }
                
                # Only append to log if signal is not UNKNOWN
                # This prevents fake entry/exit tracking from UNKNOWN signals
                if prediction['signal'] != 'UNKNOWN':
                    self.trade_log.append(log_entry)
                    self._execute_trade(prediction, log_entry)
                # UNKNOWN signals are still tracked by position state changes
                # but won't create false P&L entries in the log
        
        except Exception as e:
            print(f'Error in next(): {e}')
    
    def _execute_trade(self, prediction, log_entry):
        """
        Execute trading decision based on Markov prediction.
        
        Parameters:
        - prediction: dict with 'signal', 'up_probability', 'confidence'
        - log_entry: dict with trading context
        """
        signal = prediction['signal']
        
        # Only trade if we have high confidence and valid signal
        if signal == 'UNKNOWN':
            return
        
        # Check if we're already in a position
        if self.position:
            # Check exit conditions
            self._check_exit(log_entry)
        else:
            # Check entry conditions
            if signal == 'UP':
                self._enter_long(prediction, log_entry)
            elif signal == 'DOWN':
                self._enter_short(prediction, log_entry)
    
    def _enter_long(self, prediction, log_entry):
        """
        Enter a long position using BRACKET ORDERS.
        
        Bracket Order ensures:
        - Entry: Market order (filled next bar)
        - SL: Stop order (triggered on intra-candle wicks)
        - TP: Limit order (with 9.0 RR ratio)
        
        Position Sizing: Risk = Position_Size % of Equity
        The distance (Entry - SL) equals exactly this risk in $ terms.
        """
        try:
            # Calculate position size based on RISK PER TRADE
            equity = self.broker.getvalue()
            risk_amount = equity * self.params.position_size  # $ amount to risk
            
            # Entry price (use market on next bar)
            entry_price = self.data.close[0]
            
            # SL distance (from sl_buffer - the 95th percentile of wicks)
            sl_distance = self.sl_buffer
            self.sl_price = entry_price - sl_distance
            
            # TP distance (9.0x the SL buffer for 9:1 RR ratio)
            # This gives us a favorable risk/reward
            tp_distance = sl_distance * 9.0
            self.tp_price = entry_price + tp_distance
            
            # Calculate position size: size = risk_amount / sl_distance
            # This ensures that SL hit = exactly risk_amount loss
            size = int(risk_amount / sl_distance) if sl_distance > 0 else 0
            
            if size > 0:
                # Place bracket order
                # parentorder = self.buy() is the parent (entry)
                # psize = position size
                # exectype = Market (enter at market)
                # transmit = False (don't execute yet)
                order = self.buy(size=size, exectype=bt.Order.Market, transmit=False)
                
                # Add SL (Stop order, price below entry)
                self.sell(size=size, exectype=bt.Order.Stop, 
                         price=self.sl_price, parent=order, transmit=False)
                
                # Add TP (Limit order, price above entry)
                self.sell(size=size, exectype=bt.Order.Limit, 
                         price=self.tp_price, parent=order, transmit=True)
                
                # Store entry info for logging
                self.entry_price = entry_price
                self.entry_signal = 'LONG'
                
                print(f'\n✅ BUY BRACKET ORDER PLACED')
                print(f'   Sequence: ({log_entry["state_0"]},{log_entry["state_1"]},{log_entry["state_2"]})')
                print(f'   UP Probability: {prediction["up_probability"]:.4f}')
                print(f'   Position Size: {size} contracts')
                print(f'   Entry Price: {entry_price:.2f}')
                print(f'   SL Price: {self.sl_price:.2f} (distance: {sl_distance:.2f})')
                print(f'   TP Price: {self.tp_price:.2f} (distance: {tp_distance:.2f})')
                print(f'   Risk Amount: ${risk_amount:.2f}')
                print(f'   Risk-Reward Ratio: 1:{(tp_distance/sl_distance):.1f}')
                print()
            else:
                print(f'⚠️  SKIPPED BUY - Position size <= 0 (risk_amount={risk_amount:.2f}, sl_distance={sl_distance:.2f})')
        
        except Exception as e:
            print(f'❌ ERROR in _enter_long: {e}')
    
    def _enter_short(self, prediction, log_entry):
        """
        Enter a short position using BRACKET ORDERS.
        
        Bracket Order ensures:
        - Entry: Market order (filled next bar)
        - SL: Stop order (triggered on intra-candle wicks)
        - TP: Limit order (with 9.0 RR ratio)
        
        Position Sizing: Risk = Position_Size % of Equity
        The distance (Entry + SL) equals exactly this risk in $ terms.
        """
        try:
            # Calculate position size based on RISK PER TRADE
            equity = self.broker.getvalue()
            risk_amount = equity * self.params.position_size  # $ amount to risk
            
            # Entry price (use market on next bar)
            entry_price = self.data.close[0]
            
            # SL distance (from sl_buffer - the 95th percentile of wicks)
            sl_distance = self.sl_buffer
            self.sl_price = entry_price + sl_distance
            
            # TP distance (9.0x the SL buffer for 9:1 RR ratio)
            # This gives us a favorable risk/reward
            tp_distance = sl_distance * 9.0
            self.tp_price = entry_price - tp_distance
            
            # Calculate position size: size = risk_amount / sl_distance
            # This ensures that SL hit = exactly risk_amount loss
            size = int(risk_amount / sl_distance) if sl_distance > 0 else 0
            
            if size > 0:
                # Place bracket order
                # parentorder = self.sell() is the parent (entry)
                # psize = position size
                # exectype = Market (enter at market)
                # transmit = False (don't execute yet)
                order = self.sell(size=size, exectype=bt.Order.Market, transmit=False)
                
                # Add SL (Stop order, price above entry)
                self.buy(size=size, exectype=bt.Order.Stop, 
                        price=self.sl_price, parent=order, transmit=False)
                
                # Add TP (Limit order, price below entry)
                self.buy(size=size, exectype=bt.Order.Limit, 
                        price=self.tp_price, parent=order, transmit=True)
                
                # Store entry info for logging
                self.entry_price = entry_price
                self.entry_signal = 'SHORT'
                
                print(f'\n✅ SELL BRACKET ORDER PLACED')
                print(f'   Sequence: ({log_entry["state_0"]},{log_entry["state_1"]},{log_entry["state_2"]})')
                print(f'   UP Probability: {prediction["up_probability"]:.4f}')
                print(f'   Position Size: {size} contracts')
                print(f'   Entry Price: {entry_price:.2f}')
                print(f'   SL Price: {self.sl_price:.2f} (distance: {sl_distance:.2f})')
                print(f'   TP Price: {self.tp_price:.2f} (distance: {tp_distance:.2f})')
                print(f'   Risk Amount: ${risk_amount:.2f}')
                print(f'   Risk-Reward Ratio: 1:{(tp_distance/sl_distance):.1f}')
                print()
            else:
                print(f'⚠️  SKIPPED SELL - Position size <= 0 (risk_amount={risk_amount:.2f}, sl_distance={sl_distance:.2f})')
        
        except Exception as e:
            print(f'❌ ERROR in _enter_short: {e}')
    
    def _check_exit(self, log_entry):
        """
        Exit logic is now handled by Bracket Orders automatically.
        
        This method is kept for backward compatibility but is no longer needed.
        Bracket orders manage:
        - SL (Stop order) - triggered on intra-candle wicks
        - TP (Limit order) - triggered when price hits profit target
        
        This method is called but does nothing since bracket orders handle exits.
        """
        # Exit logic is now automatic via bracket orders
        # No manual checking needed
        pass
    
    def notify_trade(self, trade):
        """
        Called when a trade is closed.
        
        Captures detailed exit information including:
        - Entry and exit prices
        - PnL and PnL % 
        - Reason for exit (TP or SL)
        - Position duration
        """
        if not trade.isclosed:
            return
        
        # Calculate exit details
        entry_price = trade.barlen
        exit_reason = "MANUAL_CLOSE"
        
        # Determine exit reason based on orders
        if trade.barlen == 1:
            if trade.pnl > 0:
                exit_reason = "TAKE_PROFIT"
            else:
                exit_reason = "STOP_LOSS"
        
        pnl_pct = (trade.pnl / (trade.barlen * trade.price)) * 100 if trade.barlen > 0 else 0
        
        print(f'\n📊 TRADE CLOSED')
        print(f'   Entry: {trade.baropen:.2f}')
        print(f'   Exit: {trade.barclose:.2f}')
        print(f'   PnL: {trade.pnl:.2f}')
        print(f'   PnL %: {pnl_pct:.2f}%')
        print(f'   Bars: {trade.barlen}')
        print(f'   Exit Reason: {exit_reason}')
        print()
    
    def notify_order(self, order):
        """
        Called when an order status changes.
        
        Currently unused but available for future order tracking.
        """
        pass
    
    def _calculate_interest_velocity(self, df):
        """
        Calculate interest_velocity for collateral factor development.
        
        Formula: interest_velocity = interest_accrued_usd / total_collateral_usd
        
        This is a placeholder for future feature integration when collateral
        data becomes available in the CSV.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with 'interest_accrued_usd' and 'total_collateral_usd' columns
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with added 'interest_velocity' column
        """
        # Check if required columns exist
        required_cols = ['interest_accrued_usd', 'total_collateral_usd']
        
        if all(col in df.columns for col in required_cols):
            # Calculate interest velocity
            df['interest_velocity'] = (
                df['interest_accrued_usd'] / df['total_collateral_usd'].replace(0, np.nan)
            )
            # Fill NaN values with 0
            df['interest_velocity'] = df['interest_velocity'].fillna(0)
            print(f"✅ Interest velocity calculated: {df['interest_velocity'].describe()}")
        else:
            print(f"\n⚠️  COLLATERAL DATA NOT AVAILABLE")
            print(f"   Expected columns: {required_cols}")
            print(f"   Available columns: {df.columns.tolist()}")
            print(f"   Add 'interest_accrued_usd' and 'total_collateral_usd' columns to CSV for collateral factor features")
        
        return df
    
    def stop(self):
        """Called at the end of the backtest."""
        print("\n" + "="*70)
        print("MARKOV STRATEGY BACKTEST COMPLETE")
        print("="*70)
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        print(f"Total Predictions: {len(self.trade_log)}")
        print(f"\n✅ Results saved to trade_log")


# ============================================================================
# DATA FEEDER
# ============================================================================

class PandasData(bt.feeds.PandasData):
    """
    Custom PandasData feeder that handles Volatility 100 Index data.
    """
    
    params = (
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', None),
        ('openinterest', None),
    )


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_backtrader_backtest(csv_file, cash=100000, split_ratio=0.80):
    """
    Run backtrader backtest with Markov strategy using Bracket Orders.
    
    Key Features:
    - Bracket Orders for automatic SL/TP management
    - SL and TP triggered on intra-candle wicks (High/Low)
    - Risk-based position sizing (risk = position_size % of equity)
    - 9.0x Risk-Reward ratio (tp_distance = sl_distance * 9.0)
    - Confidence filtering to reduce false signals
    
    Parameters:
    - csv_file: Path to OHLC data
    - cash: Starting cash (default: 100,000)
    - split_ratio: % of data for "training" (default: 0.80)
    
    Returns:
    - cerebro: backtrader Cerebro instance with results
    - strategy: Strategy instance with trade_log
    
    Trade Logging:
    - All predictions captured (including filtered/UNKNOWN signals)
    - Position sizing calculated from risk % 
    - SL/TP calculated from dynamic sl_buffer (95th percentile of wicks)
    - Exit reason tracked (SL vs TP)
    """
    
    print("\n" + "="*70)
    print("LOADING DATA FOR BACKTRADER")
    print("="*70)
    
    # Load and prepare data
    df = pd.read_csv(csv_file, sep='\t', engine='python', on_bad_lines='skip')
    
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('<', '').str.replace('>', '').str.upper()
    df = df[['DATE', 'TIME', 'CLOSE', 'HIGH', 'LOW', 'OPEN']].copy()
    df.columns = ['DATE', 'TIME', 'close', 'high', 'low', 'open']
    
    # Create datetime index
    df['datetime'] = pd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME'].astype(str))
    df.set_index('datetime', inplace=True)
    df.drop(['DATE', 'TIME'], axis=1, inplace=True)
    
    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"✅ Loaded {len(df)} candles")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # =====================================================================
    # COLLATERAL FACTOR PLACEHOLDER
    # =====================================================================
    # To add collateral-based features, ensure your CSV includes:
    #  - 'interest_accrued_usd': Total interest accrued
    #  - 'total_collateral_usd': Total collateral value
    # Then the strategy will automatically calculate:
    #  - 'interest_velocity': interest_accrued_usd / total_collateral_usd
    # =====================================================================
    
    # Split data: use 80% for "training" and 20% for testing
    split_idx = int(len(df) * split_ratio)
    df_test = df.iloc[split_idx:].copy()
    
    print(f"\n📊 Data split:")
    print(f"   Training window: {split_idx} candles")
    print(f"   Test set: {len(df_test)} candles (used for backtest)")
    
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy
    cerebro.addstrategy(MarkovStrategy)
    
    # Add data
    data = PandasData(dataname=df_test)
    cerebro.adddata(data)
    
    # Broker settings
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.0002)  # 0.02% commission
    
    # Add analyzers for metrics
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"\n🚀 Running backtest...")
    print(f"   Initial cash: {cash:.2f}")
    print(f"   Bracket Orders: ENABLED (SL/TP managed automatically)")
    print(f"   RR Ratio: 1:9.0 (tp_distance = 9x sl_distance)")
    print(f"   Position Sizing: Risk-based (5% risk per trade)")
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Extract metrics
    final_value = cerebro.broker.getvalue()
    pnl = final_value - cash
    pnl_pct = (pnl / cash) * 100
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS SUMMARY")
    print("="*70)
    print(f"\n💰 PORTFOLIO:")
    print(f"   Starting Cash: {cash:.2f}")
    print(f"   Final Value: {final_value:.2f}")
    print(f"   Total P&L: {pnl:.2f}")
    print(f"   Return %: {pnl_pct:.2f}%")
    
    # Get analyzers results
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    
    if sharpe:
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    
    if drawdown:
        print(f"   Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A')}%")
    
    print(f"\n📈 SIGNALS:")
    print(f"   Total predictions: {len(strategy.trade_log)}")
    
    # Save results
    if len(strategy.trade_log) > 0:
        trade_df = pd.DataFrame(strategy.trade_log)
        trade_df.to_csv('backtrader_trade_log.csv', index=False)
        print(f"\n💾 Trade log saved: backtrader_trade_log.csv")
    
    return cerebro, strategy


if __name__ == "__main__":
    
    CSV_FILE = r"PATH_TO_YOUR_DATA_FILE.csv"  # Replace with your data file path
    
    # Run the backtest
    cerebro, strategy = run_backtrader_backtest(CSV_FILE, cash=100000, split_ratio=0.80)
    
    print("\n✅ BACKTEST COMPLETE")
    print("\n📝 Output files:")
    print("   - backtrader_trade_log.csv (all trades and signals)")
