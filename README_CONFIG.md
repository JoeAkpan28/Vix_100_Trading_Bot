# Configuration Setup Guide

## 🚨 IMPORTANT - CONFIGURATION REQUIRED

Before running any of these trading scripts, you **MUST** replace the placeholder values with your own configuration.

---

## 📋 Required Configuration Steps

### 1. MT5 Trading Bot Configuration (`MT5_Trading_Bot.py`)

Replace these placeholder values:

```python
# Account Configuration
ACCOUNT_ID = 0000000  # Your MT5 account ID
PASSWORD = "YOUR_PASSWORD"  # Your MT5 password
SERVER = "YOUR_BROKER_SERVER"  # Your broker server name
SYMBOL = "YOUR_SYMBOL"  # Trading symbol (e.g., "Volatility 100 Index")

# Telegram Configuration
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"  # Get from @BotFather on Telegram
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"  # Your Telegram chat ID

# Trading Parameters
INITIAL_CAPITAL = 10000.0  # Your starting capital
RISK_PERCENT = 2.0  # Risk percentage per trade
REWARD_RISK_RATIO = 9.0  # Risk/Reward ratio
CURRENCY_SYMBOL = "$"  # Your currency symbol
```

### 2. Data File Paths

Replace these file path placeholders:

- `backtrader_strategy.py`: `PATH_TO_YOUR_DATA_FILE.csv`
- `signal_decay.py`: `PATH_TO_YOUR_DATA_FILE.csv`
- `Distance_Optimizer.py`: `YOUR_BACKTEST_RESULTS_FILE.csv`

### 3. Markov Matrix Data

Ensure you have a trained `markov_brain.json` file in the same directory, or update the matrix path in the scripts.

---

## 🔧 How to Get Required Values

### MT5 Account Details
1. Open your MT5 terminal
2. Go to **Tools → Options**
3. Find your **Account ID**, **Server**, and **Password**

### Telegram Bot Setup
1. Talk to **@BotFather** on Telegram
2. Create a new bot with `/newbot`
3. Copy the **Bot Token** (looks like: `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)
4. Start a chat with your bot and send a message
5. Use **@userinfobot** to get your **Chat ID**

### Data Files
- Export historical data from your MT5 terminal
- Save as CSV with columns: DATE, TIME, OPEN, HIGH, LOW, CLOSE
- Update file paths in the scripts

---

## ⚠️ Security Notes

- **NEVER** commit your actual credentials to version control
- Use environment variables for production deployments
- Keep your `markov_brain.json` file private - it contains your trading strategy
- Test with demo accounts before using real money

---

## 🚀 Quick Start Checklist

- [ ] Replace MT5 account credentials
- [ ] Set up Telegram bot and get token/chat ID
- [ ] Update data file paths
- [ ] Verify `markov_brain.json` exists
- [ ] Test with demo account first
- [ ] Start with small position sizes

---

## 📞 Support

If you need help with configuration:
1. Check the MT5 documentation for account setup
2. Visit Telegram's Bot documentation for bot creation
3. Review the script comments for parameter explanations

**Happy Trading! 📈**
