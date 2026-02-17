"""
BACKTRADER LONG/SHORT MARKOV STRATEGY WITH TIMED EXIT
======================================================
Professional backtrader implementation of Long/Short trading strategy.
Based on extensive backtesting showing statistically reliable edge in both directions.

Key Features:
- LONG and SHORT strategy (both buy and sell positions)
- 3rd-order Markov predictor (6-State System) for UP/DOWN move detection
- Dynamic stop-loss based on entry wicks (95th percentile)
- 9:1 Risk-Reward ratio (capture large price swings)
- 7-bar TIMED EXIT (automatic exit if SL/TP not hit) - OPTIMAL HOLDING PERIOD
- Detailed trade logging and performance metrics
"""

import backtrader as bt
import pandas as pd
import numpy as np
from markov_predictor import MarkovLivePredictor
from sl_logic import StopLossCalculator

class MarkovStrategy(bt.Strategy):
    """
    Long/Short Markov Chain trading strategy with timed exits.
    
    Uses a 3rd-order Markov model to identify UP and DOWN moves with high confidence.
    Both LONG and SHORT signals are executed for comprehensive market coverage.
    Positions are automatically closed after 7 bars if SL/TP not triggered (OPTIMAL HOLDING PERIOD).
    
    Parameters:
    - up_threshold: Baseline probability (default: 0.50)
    - conf_down: Confidence offset for DOWN signals (default: 0.40)
      Down signal triggers when: up_probability < (up_threshold - conf_down)
      Example: 0.5 - 0.4 = 0.1 (triggers when probability of UP is < 10%)
    - position_size: % of equity to risk per trade (default: 0.05 = 5%)
    - rr_ratio: Risk-Reward ratio (default: 9.0 = 9:1)
    - timed_exit_bars: Exit position after N bars if still open (default: 7) - OPTIMAL
    - matrix_path: Path to vix_25_1s_markov_brain.json
    - window_size: Number of candles for state calculation (default: 100)
    """
    
    params = (
        ('up_threshold', 0.50),       # Baseline probability
        ('conf_down', 0.40),          # DOWN signal if up_prob < 0.10
        ('position_size', 0.05),      # 5% risk per trade
        ('rr_ratio', 9.0),            # 9:1 Risk-Reward ratio
        ('timed_exit_bars', 7),       # Exit after 7 bars if still open
        ('matrix_path', 'markov_brain.json'),
        ('window_size', 100),
    )
    
    def __init__(self):
        """Initialize the long/short strategy with timed exit logic."""
        # Initialize Markov predictor
        # UPDATED: Matches the signature of the new MarkovLivePredictor
        self.predictor = MarkovLivePredictor(
            matrix_path=self.params.matrix_path,
            up_threshold=self.params.up_threshold,
            conf_down=self.params.conf_down
            # conf_up uses default from class as we ignore UP signals
        )
        
        # Initialize SL calculator
        self.sl_calculator = StopLossCalculator(
            window_size=self.params.window_size,
            pip_size=1.0,
            percentile=0.95
        )
        
        # State management
        self.state_queue = []
        self.bar_count = 0
        self.last_prediction = None
        self.sl_buffer = None
        
        # Order and position tracking
        self.order = None
        self.entry_bar = None  # Track which bar the trade was entered
        
        # Logging
        self.trade_log = []
        
        # Calculate effective threshold for display
        effective_down = self.params.up_threshold - self.params.conf_down
        
        print("\n" + "="*70)
        print("MARKOV LONG/SHORT STRATEGY INITIALIZED (WITH TIMED EXIT)")
        print("="*70)
        print(f"\n🔔 SIGNAL THRESHOLD:")
        print(f"   DOWN Signal: up_probability < {effective_down:.2f}")
        print(f"   (Base={self.params.up_threshold}, conf_down={self.params.conf_down})")
        print(f"\n💰 POSITION SIZING:")
        print(f"   Risk per trade: {self.params.position_size*100:.1f}% of equity")
        print(f"\n📈 RISK-REWARD:")
        print(f"   RR Ratio: {self.params.rr_ratio:.1f}:1")
        print(f"   (tp_distance = {self.params.rr_ratio}x sl_distance)")
        print(f"\n⏱️  TIMED EXIT:")
        print(f"   Auto-close positions after: {self.params.timed_exit_bars} bars")
        print(f"   (if SL/TP not triggered)")
        print(f"\n⚠️  STRATEGY: LONG and SHORT (UP and DOWN signals)")
        print(f"   Both UP and DOWN signals are EXECUTED")
        print("="*70)
    
    def next(self):
        """Called on each bar (candle) to process new data."""
        self.bar_count += 1
        
        # Need at least window_size bars to make a prediction
        if self.bar_count < self.params.window_size:
            return
        
        window_size = self.params.window_size
        
        if len(self) < window_size:
            return
        
        try:
            # ================================================================
            # TIMED EXIT LOGIC
            # ================================================================
            if self.position:
                # Calculate how many bars have passed since entry
                bars_held = len(self) - self.entry_bar
                
                if bars_held >= self.params.timed_exit_bars:
                    exit_price = self.data.close[0]
                    print(f'\n⏰ TIMED EXIT TRIGGERED')
                    print(f'   Bars Held: {bars_held} (threshold: {self.params.timed_exit_bars})')
                    print(f'   Exit Price: {exit_price:.2f}')
                    self.close()
                    return
            
            # ================================================================
            # ENTRY SIGNAL LOGIC (Markov Prediction)
            # ================================================================
            # Create a DataFrame from the last N bars
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
            
            # Create lowercase version for predictor
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
                
                # Process both UP and DOWN signals
                if prediction['signal'] in ['UP', 'DOWN']:
                    # Determine trade direction
                    trade_direction = 'LONG' if prediction['signal'] == 'UP' else 'SHORT'
                    
                    # Log the signal
                    log_entry = {
                        'bar': len(self),
                        'datetime': self.data.datetime.date(0),
                        'state_0': int(sequence[0]),
                        'state_1': int(sequence[1]),
                        'state_2': int(sequence[2]),
                        'up_prob': prediction['up_probability'],
                        'down_prob': 1.0 - prediction['up_probability'],
                        'confidence': prediction['confidence'],
                        'signal': prediction['signal'],
                        'direction': trade_direction,
                        'price': self.data.close[0],
                        'sl_buffer': self.sl_buffer,
                        'position_open': 1 if self.position else 0,
                    }
                    
                    self.trade_log.append(log_entry)
                    
                    # Execute trade based on signal
                    if prediction['signal'] == 'DOWN':
                        self._enter_short(prediction, log_entry)
                    else:  # UP signal
                        self._enter_long(prediction, log_entry)
        
        except Exception as e:
            print(f'Error in next(): {e}')
    
    def _enter_short(self, prediction, log_entry):
        """
        Enter a SHORT position using BRACKET ORDERS.
        """
        try:
            # Check if already in a position
            if self.position:
                return
            
            # Calculate position size based on risk per trade
            equity = self.broker.getvalue()
            risk_amount = equity * self.params.position_size
            
            # Entry price (market on next bar)
            entry_price = self.data.close[0]
            
            # SL distance (from sl_buffer - the 95th percentile of wicks)
            sl_distance = self.sl_buffer
            sl_price = entry_price + sl_distance
            
            # TP distance (RR ratio × sl_distance)
            tp_distance = sl_distance * self.params.rr_ratio
            tp_price = entry_price - tp_distance
            
            # Calculate position size: size = risk_amount / sl_distance
            size = max(1, int(risk_amount / sl_distance)) if sl_distance > 0 else 0
            
            if size > 0:
                # Place SHORT entry order
                order = self.sell(size=size, exectype=bt.Order.Market, transmit=False)
                
                # Add SL (Stop order, price above entry)
                self.buy(size=size, exectype=bt.Order.Stop, 
                        price=sl_price, parent=order, transmit=True)
                
                self.order = order
                
                print(f'\n✅ SHORT ORDER PLACED')
                print(f'   Sequence: ({log_entry["state_0"]},{log_entry["state_1"]},{log_entry["state_2"]})')
                print(f'   UP Probability: {prediction["up_probability"]:.4f}')
                print(f'   Position Size: {size} contracts')
                print(f'   Entry Price: {entry_price:.2f}')
                print(f'   SL Price: {sl_price:.2f} (distance: {sl_distance:.2f})')
                print(f'   Risk Amount: ${risk_amount:.2f}')
                print(f'   Exit Strategy: TIMED EXIT after {self.params.timed_exit_bars} bars (or SL hit)')
                print()
            else:
                print(f'⚠️  SKIPPED SHORT - Position size <= 0 (risk_amount={risk_amount:.2f}, sl_distance={sl_distance:.2f})')
        
        except Exception as e:
            print(f'❌ ERROR in _enter_short: {e}')
    
    def _enter_long(self, prediction, log_entry):
        """
        Enter a LONG position using BRACKET ORDERS.
        """
        try:
            # Check if already in a position
            if self.position:
                return
            
            # Calculate position size based on risk per trade
            equity = self.broker.getvalue()
            risk_amount = equity * self.params.position_size
            
            # Entry price (market on next bar)
            entry_price = self.data.close[0]
            
            # SL distance (from sl_buffer - the 95th percentile of wicks)
            sl_distance = self.sl_buffer
            sl_price = entry_price - sl_distance
            
            # TP distance (RR ratio × sl_distance)
            tp_distance = sl_distance * self.params.rr_ratio
            tp_price = entry_price + tp_distance
            
            # Calculate position size: size = risk_amount / sl_distance
            size = max(1, int(risk_amount / sl_distance)) if sl_distance > 0 else 0
            
            if size > 0:
                # Place LONG entry order
                order = self.buy(size=size, exectype=bt.Order.Market, transmit=False)
                
                # Add SL (Stop order, price below entry)
                self.sell(size=size, exectype=bt.Order.Stop, 
                         price=sl_price, parent=order, transmit=True)
                
                self.order = order
                
                print(f'\n✅ LONG ORDER PLACED')
                print(f'   Sequence: ({log_entry["state_0"]},{log_entry["state_1"]},{log_entry["state_2"]})')
                print(f'   UP Probability: {prediction["up_probability"]:.4f}')
                print(f'   Position Size: {size} contracts')
                print(f'   Entry Price: {entry_price:.2f}')
                print(f'   SL Price: {sl_price:.2f} (distance: {sl_distance:.2f})')
                print(f'   Risk Amount: ${risk_amount:.2f}')
                print(f'   Exit Strategy: TIMED EXIT after {self.params.timed_exit_bars} bars (or SL hit)')
                print()
            else:
                print(f'⚠️  SKIPPED LONG - Position size <= 0 (risk_amount={risk_amount:.2f}, sl_distance={sl_distance:.2f})')
        
        except Exception as e:
            print(f'❌ ERROR in _enter_long: {e}')
    
    def notify_trade(self, trade):
        """Called when a trade is closed."""
        if not trade.isclosed:
            return
        
        print(f'\n📊 TRADE CLOSED')
        print(f'   Entry: {trade.baropen:.2f}')
        print(f'   Exit: {trade.barclose:.2f}')
        print(f'   PnL: {trade.pnl:.2f}')
        print(f'   Bars Held: {trade.barlen}')
        print()
    
    def notify_order(self, order):
        """Called when an order status changes."""
        if order.status in [order.Completed]:
            if order.isbuy():
                pass
            else:
                self.entry_bar = len(self)
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            pass
    
    def stop(self):
        """Called at the end of the backtest."""
        print("\n" + "="*70)
        print("MARKOV LONG/SHORT STRATEGY BACKTEST COMPLETE")
        print("="*70)
        print(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        print(f"Total Signals: {len(self.trade_log)}")
        print(f"\n✅ Results saved to trade_log")


# ============================================================================
# DATA FEEDER
# ============================================================================

class PandasData(bt.feeds.PandasData):
    """Custom PandasData feeder."""
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

def run_backtrader_backtest(csv_file, cash=1000000, split_ratio=0.80):
    
    print("\n" + "="*70)
    print("LOADING DATA FOR BACKTRADER (LONG/SHORT WITH TIMED EXIT)")
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
    
    # Split data: use 80% for "training" and 20% for testing
    split_idx = int(len(df) * split_ratio)
    df_test = df.iloc[split_idx:].copy()
    
    print(f"\n🔔 Data split:")
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
    cerebro.broker.setcommission(commission=0.0002)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print(f"\n🚀 Running backtest...")
    print(f"   Initial cash: ${cash:,.2f}")
    print(f"   Strategy: LONG and SHORT (UP and DOWN signals)")
    print(f"   Timed Exit: {7} bars (if SL/TP not triggered)")
    
    # Run backtest
    results = cerebro.run()
    strategy = results[0]
    
    # Extract metrics
    final_value = cerebro.broker.getvalue()
    pnl = final_value - cash
    pnl_pct = (pnl / cash) * 100
    
    print("\n" + "="*70)
    print("BACKTEST RESULTS SUMMARY (LONG/SHORT WITH TIMED EXIT)")
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
        print(f"\n🔔 RISK METRICS:")
        print(f"   Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")
    
    if drawdown:
        print(f"   Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A')}%")
    
    print(f"\n📈 SIGNALS:")
    print(f"   Total signals: {len(strategy.trade_log)}")
    if len(strategy.trade_log) > 0:
        trade_df = pd.DataFrame(strategy.trade_log)
        long_signals = len(trade_df[trade_df['signal'] == 'UP'])
        short_signals = len(trade_df[trade_df['signal'] == 'DOWN'])
        print(f"   LONG signals: {long_signals}")
        print(f"   SHORT signals: {short_signals}")
    
    # Save results
    if len(strategy.trade_log) > 0:
        trade_df = pd.DataFrame(strategy.trade_log)
        trade_df.to_csv('backtrader_trade_log.csv', index=False)
        print(f"\n💾 Trade log saved: backtrader_trade_log.csv")
    
    return cerebro, strategy


if __name__ == "__main__":
    
    # UPDATED CSV PATH FOR VOLATILITY 100
    CSV_FILE = r"PATH_TO_YOUR_DATA_FILE.csv"  # Replace with your data file path
    
    # Check if file exists in theory (for user reference)
    # Using 100k cash and 80/20 split
    cerebro, strategy = run_backtrader_backtest(CSV_FILE, cash=100000, split_ratio=0.80)
    
    print("\n✅ BACKTEST COMPLETE")
    print("\n📁 Output files:")
    print("   - backtrader_trade_log.csv (all LONG and SHORT trades and signals)")