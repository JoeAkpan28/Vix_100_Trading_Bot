"""
MT5 Trading Bot - Hardened for 24/7 VPS Deployment

A robust trading bot for MetaTrader 5 with advanced error handling,
dynamic filling mode detection, and remote monitoring capabilities.

Features:
- Markov chain-based market prediction
- Dynamic stop loss calculation using 95th percentile wicks
- Pre-trade margin validation
- Telegram notifications for monitoring
- Automatic reconnection handling
- Server time synchronization to avoid timezone issues
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
import time
import sys
import numpy as np
import logging
import requests
import os
from typing import Dict, List, Tuple, Optional

# Configuration - REPLACE WITH YOUR OWN VALUES
ACCOUNT_ID = 0000000  # Your MT5 account ID
PASSWORD = "YOUR_PASSWORD"  # Your MT5 password
SERVER = "YOUR_BROKER_SERVER"  # Your broker server name
SYMBOL = "YOUR_SYMBOL"  # Trading symbol (e.g., "Volatility 100 Index")
TIMEFRAME = mt5.TIMEFRAME_H1  # Timeframe for trading
MAGIC_NUMBER = 123456  # Unique identifier for your trades

# Telegram Configuration - REPLACE WITH YOUR OWN VALUES
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"  # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"  # Your Telegram chat ID

# Trading Parameters - ADJUST TO YOUR RISK PREFERENCES
INITIAL_CAPITAL = 10000.0  # Starting capital in your currency
RISK_PERCENT = 2.0  # Risk percentage per trade (e.g., 2.0 = 2%)
REWARD_RISK_RATIO = 9.0  # Risk/Reward ratio (e.g., 9.0 = 9:1)
CURRENCY_SYMBOL = "$"  # Your currency symbol (e.g., "$", "€", "£")

# Stop Loss Parameters - ADJUST TO YOUR STRATEGY
SL_WINDOW_SIZE = 100  # Window size for stop loss calculation
SL_PERCENTILE = 0.95  # Percentile for stop loss buffer (0.95 = 95th percentile)

# Hardening Parameters - ADJUST AS NEEDED
NEW_CANDLE_BUFFER_SECONDS = 5  # Wait time after new candle (seconds)

# Markov Transition Matrix - LOAD FROM YOUR TRAINED MODEL
# This should be loaded from your trained markov_brain.json file
MARKOV_MATRIX = {
    "(0,0,0)": {"0": 0.4285714285714286, "1": 0.0, "2": 0.0, "3": 0.2857142857142857, "4": 0.14285714285714285, "5": 0.14285714285714285},
    "(0,0,2)": {"0": 0.3014705882352941, "1": 0.0808823529411764, "2": 0.0073529411764705, "3": 0.375, "4": 0.1544117647058824, "5": 0.0808823529411764},
    "(5,5,5)": {"0": 0.1761363636363636, "1": 0.0625, "2": 0.0056818181818181, "3": 0.4034090909090909, "4": 0.2329545454545454, "5": 0.1193181818181818},
    "(<NA>,<NA>,3)": {"0": 0.0625, "1": 0.0625, "2": 0.0625, "3": 0.0625, "4": 0.0625, "5": 0.6875}
}


def setup_logging():
    """Initialize logging configuration for file and console output."""
    logger = logging.getLogger("vix_markov_bot_hardened")
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler('vix_production_hardened_log.txt', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()
logger.info("="*70)
logger.info("🔧 HARDENED BOT INITIALIZATION (24/7 VPS PRODUCTION)")
logger.info("="*70)


class StopLossCalculator:
    """
    Calculates dynamic stop loss levels based on historical wick data.
    
    Uses the 95th percentile of entry wicks over a rolling window to determine
    optimal stop loss placement that accounts for market volatility.
    """
    
    def __init__(self, window_size=100, pip_size=1.0, percentile=0.95):
        self.window_size = window_size
        self.pip_size = pip_size
        self.percentile = percentile
        logger.info(f"✅ StopLossCalculator: window={window_size}, percentile={percentile}")
        
    def calculate_entry_wick(self, df, open_col='open', close_col='close', 
                           high_col='high', low_col='low'):
        """Calculate the wick size from the entry price of each candle."""
        df = df.copy()
        df['is_bullish'] = df[close_col] >= df[open_col]
        df['entry_wick'] = np.where(
            df['is_bullish'],
            df[open_col] - df[low_col],  # Bullish: lower wick from open
            df[high_col] - df[open_col]   # Bearish: upper wick from open
        )
        return df
    
    def calculate_sl_buffer(self, df, entry_wick_col='entry_wick'):
        """Calculate rolling stop loss buffer using percentile of wicks."""
        df = df.copy()
        df['sl_buffer'] = df[entry_wick_col].rolling(
            window=self.window_size
        ).quantile(self.percentile)
        df['sl_buffer'] = df['sl_buffer'].ffill().bfill()
        return df
    
    def calculate_sl_price(self, df, signal_col='signal', 
                          open_col='open', sl_buffer_col='sl_buffer'):
        """Calculate final stop loss price based on signal direction."""
        df = df.copy()
        df['sl_price'] = np.where(
            df[signal_col] == 1,  # Buy signal
            df[open_col] - df[sl_buffer_col],  # SL below open
            np.where(
                df[signal_col] == -1,  # Sell signal
                df[open_col] + df[sl_buffer_col],  # SL above open
                df[open_col]  # No signal
            )
        )
        return df
    
    def calculate_all(self, df, open_col='open', close_col='close',
                     high_col='high', low_col='low', signal_col='signal'):
        """Execute complete stop loss calculation pipeline."""
        df = self.calculate_entry_wick(df, open_col, close_col, high_col, low_col)
        df = self.calculate_sl_buffer(df, entry_wick_col='entry_wick')
        df = self.calculate_sl_price(df, signal_col, open_col, sl_buffer_col='sl_buffer')
        return df


class TelegramNotifier:
    """
    Handles Telegram notifications for trading events and error monitoring.
    
    Provides real-time alerts for trade entries, exits, errors, and system status.
    Essential for remote monitoring of VPS-deployed trading bots.
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)
        
        if self.enabled:
            logger.info(f"✅ Telegram notifications ENABLED")
        else:
            logger.info(f"⚠️  Telegram notifications DISABLED")
    
    def test_connection(self) -> bool:
        """Test Telegram bot connection and validity."""
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get('ok'):
                logger.info(f"✅ Telegram bot verified")
                return True
            else:
                logger.error(f"❌ Telegram error: {data}")
                return False
        except Exception as e:
            logger.error(f"❌ Telegram test failed: {e}")
            return False
    
    def send_message(self, message: str) -> bool:
        """Send a message to Telegram chat."""
        if not self.enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload, timeout=10)
            data = response.json()
            return data.get('ok', False)
        except Exception as e:
            logger.warning(f"Failed to send Telegram: {e}")
            return False
    
    def notify_order_error(self, error_type: str, retcode: int, comment: str, 
                          direction: str = "", price: float = 0):
        """
        Log order errors with retcode for remote debugging via Telegram.
        
        Common retcodes:
        - 10019: NO MONEY
        - 10016: INVALID VOLUME
        - 10017: INVALID PRICE
        - 10014: INVALID REQUEST
        - 10013: UNKNOWN SYMBOL
        - 10020: PERMISSION DENIED
        """
        message = f"""
⚠️ <b>ORDER ERROR</b>

Type: {error_type}
Retcode: {retcode}
Comment: {comment}
{f'Direction: {direction}' if direction else ''}
{f'Price: {CURRENCY_SYMBOL}{price:.5f}' if price else ''}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
        logger.warning(f"ORDER ERROR | Retcode: {retcode} | {comment}")
    
    def notify_entry(self, direction: str, entry_price: float, sl: float, 
                    tp: float, size: int, risk: float, probability: float, ticket_id: int = None):
        """Send trade entry notification."""
        emoji = "🚀" if direction == "BUY" else "📉"
        message = f"""
{emoji} <b>{direction} ENTRY</b>

📊 Entry: {entry_price:.5f}
🛑 SL: ${sl:.5f}
🎯 TP: ${tp:.5f}
📦 Size: {size} lots
💵 Risk: {CURRENCY_SYMBOL}{risk:.2f}
📈 Probability: {probability:.1%}
{f'🎫 Ticket: {ticket_id}' if ticket_id else ''}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_bot_start(self):
        """Send bot startup notification."""
        message = f"""
🤖 <b>BOT STARTED (HARDENED)</b>

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Symbol: {SYMBOL}
💰 Capital: {CURRENCY_SYMBOL}{INITIAL_CAPITAL:.2f}
🛡️ Safety: Pre-trade margin check + Dynamic filling mode
"""
        self.send_message(message)
    
    def notify_exit(self, direction: str, entry_price: float, exit_price: float,
                   reason: str, pnl: float, pnl_pct: float, ticket_id: int = None):
        """Send trade exit notification."""
        emoji = "✅" if pnl > 0 else "❌"
        message = f"""
{emoji} <b>EXIT: {reason}</b>

📊 {direction} Position
💥 Entry: {CURRENCY_SYMBOL}{entry_price:.5f}
💰 Exit: {CURRENCY_SYMBOL}{exit_price:.5f}
📈 P&L: {CURRENCY_SYMBOL}{pnl:.2f} ({pnl_pct:+.2f}%)
{f'🎫 Ticket: {ticket_id}' if ticket_id else ''}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)
    
    def notify_critical_error(self, error_message: str):
        """Send critical error notification."""
        message = f"""
🚨 <b>CRITICAL ERROR</b>

Error: {error_message}
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        self.send_message(message)


class MT5Watchdog:
    """
    Manages MT5 connection with automatic reconnection capabilities.
    
    Monitors connection health and attempts reconnection with exponential backoff
    to ensure continuous operation during network issues or server restarts.
    """
    
    def __init__(self, account_id: int, password: str, server: str, max_reconnect_attempts: int = 3):
        self.account_id = account_id
        self.password = password
        self.server = server
        self.max_reconnect_attempts = max_reconnect_attempts
        self.connected = False
        self.reconnect_attempts = 0
    
    def initialize(self) -> bool:
        """Initialize MT5 connection and authenticate."""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed, error code = {mt5.last_error()}")
                return False
            
            authorized = mt5.login(self.account_id, password=self.password, server=self.server)
            
            if authorized:
                account_info = mt5.account_info()
                if account_info is not None:
                    logger.info("="*70)
                    logger.info("✅ MT5 CONNECTION SUCCESSFUL")
                    logger.info("="*70)
                    logger.info(f"Account: {account_info.login}")
                    logger.info(f"Balance: ${account_info.balance:.2f}")
                    logger.info(f"Equity: ${account_info.equity:.2f}")
                    self.connected = True
                    self.reconnect_attempts = 0
                    return True
                else:
                    logger.error(f"Failed to get account info")
                    return False
            else:
                logger.error(f"Failed to login to MT5")
                return False
        except Exception as e:
            logger.error(f"Exception during MT5 initialization: {e}")
            return False
    
    def check_connection(self) -> bool:
        """Check if MT5 connection is still active."""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                self.connected = False
                return False
            return terminal_info.connected
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            self.connected = False
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5 with exponential backoff."""
        logger.warning("⚠️  MT5 connection lost. Attempting to reconnect...")
        
        for attempt in range(1, self.max_reconnect_attempts + 1):
            logger.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts}")
            
            try:
                mt5.shutdown()
            except:
                pass
            
            time.sleep(2 ** attempt)
            
            if self.initialize():
                logger.info(f"✅ Reconnection successful")
                return True
        
        logger.error(f"❌ Failed to reconnect")
        return False
    
    def shutdown(self):
        """Safely shutdown MT5 connection."""
        try:
            mt5.shutdown()
            logger.info("MT5 shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


class MarkovLivePredictor:
    """
    Markov chain-based market direction predictor.
    
    Uses historical state transitions to predict probability of upward movement
    based on the last 3 market states. States are determined by return quantiles.
    """
    
    def __init__(self, matrix: Dict = None, up_threshold: float = 0.50, 
                 conf_up: float = 0.15, conf_down: float = 0.40):
        self.matrix = matrix or MARKOV_MATRIX
        self.up_threshold = up_threshold
        self.threshold_up = self.up_threshold + conf_up
        self.threshold_down = self.up_threshold - conf_down
        self.up_states = ['4', '5']
        self.state_history = []
        logger.info(f"✅ MarkovLivePredictor initialized")
    
    def get_current_state(self, window_df: pd.DataFrame) -> Optional[float]:
        """Determine current market state based on return quantiles."""
        if len(window_df) < 2:
            return None
        returns = np.log(window_df['close'] / window_df['close'].shift(1))
        curr_ret = returns.iloc[-1]
        q20, q40, q50, q60, q80 = returns.quantile([0.20, 0.40, 0.50, 0.60, 0.80])
        
        if curr_ret < q20: state = 0.0
        elif curr_ret < q40: state = 1.0
        elif curr_ret < q50: state = 2.0
        elif curr_ret < q60: state = 3.0
        elif curr_ret < q80: state = 4.0
        else: state = 5.0
        return state
    
    def add_state(self, state: float) -> None:
        """Add new state to history, maintaining last 3 states."""
        self.state_history.append(state)
        if len(self.state_history) > 3:
            self.state_history = self.state_history[-3:]
    
    def predict(self) -> Dict:
        """Generate trading signal based on Markov chain prediction."""
        if len(self.state_history) < 3:
            return {"signal": "UNKNOWN", "up_probability": 0.5, "confidence": 0.0}
        
        seq_key = f"({int(self.state_history[-3])},{int(self.state_history[-2])},{int(self.state_history[-1])})"
        if seq_key not in self.matrix:
            return {"signal": "UNKNOWN", "up_probability": 0.5, "confidence": 0.0}
        
        probs = self.matrix[seq_key]
        up_prob = sum(probs.get(state, 0.0) for state in self.up_states)
        confidence = abs(up_prob - 0.5)
        
        if up_prob >= self.threshold_up:
            signal = "UP"
        elif up_prob < self.threshold_down:
            signal = "DOWN"
        else:
            signal = "UNKNOWN"
        
        return {"signal": signal, "up_probability": up_prob, "confidence": confidence}


def get_mt5_server_time() -> Optional[datetime]:
    """
    Get current server time from MT5 tick data.
    
    This avoids timezone issues when comparing position entry time
    with current time in the 7-bar exit rule.
    
    Returns:
    - Datetime of server time, or None if failed
    """
    try:
        tick = mt5.symbol_info_tick(SYMBOL)
        if tick is None:
            logger.warning("Failed to get tick data for server time")
            return None
        
        server_time = pd.to_datetime(tick.time, unit='s')
        return server_time
    
    except Exception as e:
        logger.error(f"Error getting server time: {e}")
        return None


def get_filling_mode(symbol: str) -> int:
    """
    Auto-detect broker's supported filling mode for a symbol.
    
    Deriv often throws "Unsupported filling mode" if we hardcode IOC.
    This function queries the symbol info and picks the best available mode.
    
    Parameters:
    - symbol: Trading symbol
    
    Returns:
    - Filling mode constant (FOK, IOC, or RETURN)
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.warning(f"Symbol info not found for {symbol}, defaulting to FOK")
            return mt5.ORDER_FILLING_FOK
        
        filling_mode = symbol_info.filling_mode
        
        supports_fok = filling_mode & 1  # Check if FOK (bit 0) is set
        supports_ioc = filling_mode & 2  # Check if IOC (bit 1) is set
        supports_return = filling_mode & 4  # Check if RETURN (bit 2) is set
        
        if supports_fok:
            logger.info(f"🔧 Filling Mode: FOK (Fill or Kill) - preferred")
            return mt5.ORDER_FILLING_FOK
        elif supports_ioc:
            logger.info(f"🔧 Filling Mode: IOC (Immediate or Cancel)")
            return mt5.ORDER_FILLING_IOC
        elif supports_return:
            logger.info(f"🔧 Filling Mode: RETURN (Partial Fill)")
            return mt5.ORDER_FILLING_RETURN
        else:
            logger.warning(f"No filling modes detected, defaulting to FOK")
            return mt5.ORDER_FILLING_FOK
    
    except Exception as e:
        logger.error(f"Error detecting filling mode: {e}, defaulting to FOK")
        return mt5.ORDER_FILLING_FOK


def get_market_data(symbol: str, n_candles: int, timeframe: int) -> Optional[pd.DataFrame]:
    """Fetch market data from MT5 and return as DataFrame."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get rates for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df.columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df = df.sort_values('time').reset_index(drop=True)
        return df
    
    except Exception as e:
        logger.error(f"Exception in get_market_data: {e}")
        return None


def warm_up_predictor(symbol: str, n_candles: int = 100) -> bool:
    """Initialize predictor with historical data."""
    logger.info("="*70)
    logger.info("🔥 STARTING PREDICTOR WARM-UP")
    logger.info("="*70)
    
    historical_data = get_market_data(symbol, n_candles, TIMEFRAME)
    if historical_data is None or len(historical_data) < 10:
        logger.error(f"Failed to fetch sufficient historical data")
        return False
    
    logger.info(f"Fetched {len(historical_data)} historical candles")
    predictor.state_history = []
    
    for idx in range(len(historical_data)):
        if idx < 20:
            continue
        window = historical_data.iloc[max(0, idx-99):idx+1].copy()
        state = predictor.get_current_state(window)
        if state is not None:
            predictor.add_state(state)
    
    if len(predictor.state_history) >= 3:
        logger.info(f"✅ WARM-UP COMPLETE")
        return True
    else:
        logger.error(f"Warm-up failed")
        return False


def calculate_dynamic_lot_size(sl_price: float, symbol: str) -> float:
    """Calculate position size based on risk percentage and stop loss distance."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found")
            return 0.01
        
        account_info = mt5.account_info()
        if account_info is None:
            return 0.01
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return 0.01
        
        current_price = tick.mid if hasattr(tick, 'mid') else (tick.bid + tick.ask) / 2
        balance = account_info.balance
        risk_amount = balance * (RISK_PERCENT / 100)
        
        sl_distance = abs(current_price - sl_price)
        if sl_distance > 0:
            lot_size = max(0.01, risk_amount / (sl_distance * symbol_info.trade_contract_size))
            lot_size = round(lot_size, 2)
        else:
            lot_size = 0.01
        
        return lot_size
    except Exception as e:
        logger.error(f"Error calculating lot size: {e}")
        return 0.01


def get_trade_specs(symbol: str, direction: str, df: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    """Calculate trade specifications including lot size, SL, and TP."""
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        
        current_price = tick.mid if hasattr(tick, 'mid') else (tick.bid + tick.ask) / 2
        
        df_copy = df.copy()
        df_copy['signal'] = 1 if direction == "BUY" else -1
        df_copy = sl_calculator.calculate_entry_wick(df_copy)
        df_copy = sl_calculator.calculate_sl_buffer(df_copy)
        latest_sl_buffer = df_copy['sl_buffer'].iloc[-1]
        
        logger.info(f"   SL buffer (95% percentile): {latest_sl_buffer:.5f}")
        
        if direction == "BUY":
            sl_price = current_price - latest_sl_buffer
            tp_price = current_price + (REWARD_RISK_RATIO * (current_price - sl_price))
        else:
            sl_price = current_price + latest_sl_buffer
            tp_price = current_price - (REWARD_RISK_RATIO * (sl_price - current_price))
        
        lot_size = calculate_dynamic_lot_size(sl_price, symbol)
        return (lot_size, sl_price, tp_price)
    
    except Exception as e:
        logger.error(f"Error calculating trade specs: {e}")
        return None


def place_trade(symbol: str, direction: str, lot_size: float, sl_price: float, tp_price: float) -> Optional[dict]:
    """
    Place a trade with pre-trade validation (order_check) and robust error logging.
    
    FIX #4: Use order_check() to validate margin & volume BEFORE sending
    FIX #5: Log all retcodes to Telegram for remote debugging
    """
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found")
            return None
        
        if not symbol_info.visible:
            logger.error(f"Symbol {symbol} is not visible")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Tick data not available")
            return None
        
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": float(price),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "deviation": 20,
            "magic": int(MAGIC_NUMBER),
            "comment": "VIX Markov + Dynamic SL",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": dynamic_filling_mode,
        }
        
        logger.info(f"🔍 Pre-trade validation (order_check)...")
        check_result = mt5.order_check(request)
        
        if check_result is None:
            logger.error(f"❌ order_check returned None")
            return None
        
        if check_result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = check_result.comment if hasattr(check_result, 'comment') else "Unknown error"
            logger.warning(f"❌ Pre-trade validation failed")
            logger.warning(f"   Retcode: {check_result.retcode}")
            logger.warning(f"   Comment: {error_msg}")
            
            notifier.notify_order_error(
                error_type="PRE-TRADE CHECK FAILED",
                retcode=check_result.retcode,
                comment=error_msg,
                direction=direction,
                price=price
            )
            
            return None
        
        logger.info(f"✅ Pre-trade validation passed, sending order...")
        
        result = mt5.order_send(request)
        
        if result is None:
            logger.error(f"❌ order_send returned None")
            notifier.notify_order_error(
                error_type="ORDER SEND FAILED",
                retcode=0,
                comment="order_send returned None",
                direction=direction
            )
            return None
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            error_msg = result.comment if hasattr(result, 'comment') else "Unknown error"
            logger.warning(f"❌ Order execution failed")
            logger.warning(f"   Retcode: {result.retcode}")
            logger.warning(f"   Comment: {error_msg}")
            
            notifier.notify_order_error(
                error_type="ORDER EXECUTION FAILED",
                retcode=result.retcode,
                comment=error_msg,
                direction=direction,
                price=price
            )
            return None
        
        logger.info(f"✅ {direction} EXECUTED")
        logger.info(f"   Ticket: {result.order}")
        logger.info(f"   Entry: ${price:.5f}")
        logger.info(f"   SL: ${sl_price:.5f}")
        logger.info(f"   TP: ${tp_price:.5f}")
        logger.info(f"   Lot: {lot_size}")
        
        prediction = predictor.predict()
        account = mt5.account_info()
        risk_amount = account.balance * (RISK_PERCENT / 100)
        notifier.notify_entry(direction, price, sl_price, tp_price, lot_size, risk_amount, 
                             prediction['up_probability'], result.order)
        
        return {'ticket': result.order, 'price': price}
    
    except Exception as e:
        logger.error(f"Exception in place_trade: {e}")
        notifier.notify_critical_error(f"place_trade exception: {str(e)[:100]}")
        return None


def get_open_positions(symbol: str) -> list:
    """Get list of open positions for the specified symbol."""
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None:
            return []
        return list(positions)
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        return []


def should_exit_position(position, max_bars: int = 7) -> bool:
    """
    Check if position should exit (7-bar rule) using MT5 SERVER TIME.
    
    FIX #1: Now uses get_mt5_server_time() instead of local PC time
    This avoids timezone discrepancies on VPS.
    """
    try:
        server_time = get_mt5_server_time()
        if server_time is None:
            logger.warning("Could not get server time, skipping exit check")
            return False
        
        entry_time = pd.to_datetime(position.time, unit='s')
        time_diff = server_time - entry_time
        
        bars_held = time_diff.total_seconds() / 3600
        
        should_exit = bars_held >= max_bars
        
        if should_exit:
            logger.info(f"Position held for {bars_held:.1f} hours (>= {max_bars} bars) - EXITING")
        
        return should_exit
    
    except Exception as e:
        logger.error(f"Error in should_exit_position: {e}")
        return False


def close_position(ticket_id: int, reason: str = "Manual Close") -> bool:
    """Close an open position by ticket ID."""
    try:
        position = mt5.positions_get(ticket=ticket_id)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket_id} not found")
            return False
        
        pos = position[0]
        symbol = pos.symbol
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket_id,
            "price": float(price),
            "deviation": 20,
            "magic": int(MAGIC_NUMBER),
            "comment": f"Close - {reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": dynamic_filling_mode,
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.comment if result else 'None'}")
            return False
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            pnl = (price - pos.price_open) * pos.volume
        else:
            pnl = (pos.price_open - price) * pos.volume
        
        pnl_pct = ((price - pos.price_open) / pos.price_open) * 100 if pos.price_open != 0 else 0
        
        logger.info(f"✅ Position {ticket_id} closed | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")
        
        direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        notifier.notify_exit(direction, pos.price_open, price, reason, pnl, pnl_pct, ticket_id)
        
        return True
    
    except Exception as e:
        logger.error(f"Exception in close_position: {e}")
        return False


def trading_loop(sleep_interval: int = 60):
    """
    Main trading loop with all hardening fixes applied.
    
    HARDENING FIXES:
    1. Uses MT5 server time for 7-bar rule
    2. Dynamic filling mode detection
    3. 5-second buffer after new candle
    4. Pre-trade order_check() validation
    5. All retcodes logged to Telegram
    """
    logger.info("="*70)
    logger.info("🚀 STARTING HARDENED TRADING LOOP (24/7 VPS)")
    logger.info("="*70)
    logger.info(f"Sleep interval: {sleep_interval}s")
    logger.info(f"New candle buffer: {NEW_CANDLE_BUFFER_SECONDS}s")
    logger.info(f"Server time mode: Enabled (no timezone issues)")
    logger.info(f"Pre-trade check: Enabled (order_check validation)")
    logger.info(f"Dynamic filling: Enabled (auto-detect)")
    logger.info("="*70)
    
    notifier.notify_bot_start()
    
    last_candle_time = None
    bar_count = 0
    
    try:
        while True:
            if not watchdog.check_connection():
                if not watchdog.reconnect():
                    logger.error("Cannot reconnect to MT5")
                    break
            
            df = get_market_data(SYMBOL, 100, TIMEFRAME)
            if df is None or len(df) != 100:
                logger.debug("Data not ready")
                time.sleep(sleep_interval)
                continue
            
            latest_candle_time = df['time'].iloc[-1]
            if last_candle_time is not None and latest_candle_time <= last_candle_time:
                time.sleep(sleep_interval)
                continue
            
            logger.info(f"📊 NEW CANDLE DETECTED, waiting {NEW_CANDLE_BUFFER_SECONDS}s for stability...")
            time.sleep(NEW_CANDLE_BUFFER_SECONDS)
            
            df = get_market_data(SYMBOL, 100, TIMEFRAME)
            if df is None or len(df) != 100:
                logger.warning("Data refresh failed after buffer")
                continue
            
            last_candle_time = df['time'].iloc[-1]
            bar_count += 1
            
            logger.info(f"{'-'*70}")
            logger.info(f"📊 NEW CANDLE #{bar_count} | {last_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'-'*70}")
            
            state = predictor.get_current_state(df)
            if state is None:
                logger.warning("State not available")
                time.sleep(sleep_interval)
                continue
            
            predictor.add_state(state)
            
            signal_result = predictor.predict()
            signal = signal_result['signal']
            
            logger.info(f"State: {state} | History: {predictor.state_history}")
            logger.info(f"Signal: {signal} | Prob: {signal_result['up_probability']:.2%}")
            
            open_positions = get_open_positions(SYMBOL)
            
            if len(open_positions) == 0:
                if signal in ("UP", "DOWN"):
                    logger.info(f"✅ SIGNAL DETECTED: {signal}")
                    
                    specs = get_trade_specs(SYMBOL, signal, df)
                    if specs is None:
                        logger.warning("Trade specs unavailable")
                        time.sleep(sleep_interval)
                        continue
                    
                    lot_size, sl_price, tp_price = specs
                    direction = "BUY" if signal == "UP" else "SELL"
                    
                    logger.info(f"📍 TRADE ENTRY (Hardened):")
                    logger.info(f"   Direction: {direction}")
                    logger.info(f"   Lot: {lot_size}")
                    logger.info(f"   SL: ${sl_price:.5f}")
                    logger.info(f"   TP: ${tp_price:.5f}")
                    
                    result = place_trade(SYMBOL, direction, lot_size, sl_price, tp_price)
                    if result is None:
                        logger.warning("Trade placement failed")
                    else:
                        logger.info(f"✅ Trade placed - Ticket: {result['ticket']}")
                else:
                    logger.info(f"Signal: {signal} (no action)")
            else:
                logger.info(f"📌 {len(open_positions)} open position(s)")
                
                for position in open_positions:
                    if should_exit_position(position, max_bars=7):
                        logger.info(f"⏱️  Position {position.ticket}: 7-bar exit - CLOSING")
                        close_position(position.ticket, reason="7-bar exit rule")
                    else:
                        logger.info(f"   Ticket: {position.ticket} | Entry: ${position.price_open:.5f} | Current: ${position.price_current:.5f}")
            
            time.sleep(sleep_interval)
    
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        notifier.send_message("⏹️ Bot stopped by user")
    
    except Exception as e:
        logger.error(f"Trading loop crashed: {e}", exc_info=True)
        notifier.notify_critical_error(f"Loop crash: {str(e)[:100]}")
    
    finally:
        logger.info("="*70)
        logger.info("🔌 SHUTTING DOWN")
        logger.info("="*70)
        watchdog.shutdown()
        logger.info("✅ Shutdown complete")


if __name__ == "__main__":
    sl_calculator = StopLossCalculator(window_size=SL_WINDOW_SIZE, percentile=SL_PERCENTILE)
    predictor = MarkovLivePredictor()
    notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    watchdog = MT5Watchdog(ACCOUNT_ID, PASSWORD, SERVER)
    
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        notifier.test_connection()
    
    if not watchdog.initialize():
        logger.critical("❌ FAILED TO INITIALIZE MT5")
        raise Exception("MT5 initialization failed")
    
    dynamic_filling_mode = get_filling_mode(SYMBOL)
    logger.info(f"✅ Dynamic filling mode initialized for {SYMBOL}")
    
    if not warm_up_predictor(SYMBOL, n_candles=100):
        logger.warning("⚠️  Warm-up incomplete, but continuing")
    
    logger.info("✅ All systems initialized. Starting trading loop...")
    trading_loop(sleep_interval=60)
