# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Binance Futures trading bot written in Python that implements a scalping strategy. The bot places limit orders at random price levels and automatically closes positions when orders are filled. It includes a real-time monitoring dashboard and Telegram notifications.

## Development Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Binance API credentials
```

### Running the Bot
```bash
# Basic usage with ETH/USDC
python runbot.py --symbol ETHUSDC --quantity 0.2 --take_profit 0.5 --max-orders 50 --wait-time 30

# BTC/USDC example
python runbot.py --symbol BTCUSDC --quantity 0.001 --take_profit 50 --max-orders 50 --wait-time 30
```

### Dashboard
```bash
# Start monitoring dashboard
python dashboard.py

# Monitor specific symbol with custom refresh rate
python dashboard.py --symbol BTCUSDC --refresh-rate 10
```

### Telegram Bot
```bash
# Run Telegram bot for account notifications
python telegram_bot.py
```

### Account Info
```bash
# Get current account information
python get_account_info.py
```

## Architecture

### Core Components

- **runbot.py**: Main trading bot with `TradingBot` class that handles:
  - Order placement using POST-ONLY limit orders
  - WebSocket monitoring for real-time order status
  - Automatic closing order placement when orders fill
  - Risk management with configurable limits

- **dashboard.py**: Real-time monitoring interface (`TradingDashboard` class) featuring:
  - Live account balance and PnL tracking
  - Position monitoring with entry/mark prices
  - Open orders management
  - Trading statistics and recent trades display

- **telegram_bot.py**: Notification system (`BinanceAccountAnalyzer` class) providing:
  - Hourly account overview reports
  - Transaction log analysis
  - PnL tracking and alerts

- **get_account_info.py**: Utility for account information retrieval

### Key Classes and Flow

1. **TradingConfig**: Dataclass managing trading parameters (symbol, quantity, take_profit, etc.)
2. **OrderMonitor**: Thread-safe order state tracking
3. **TradingBot**: Main bot logic coordinating order placement and monitoring
4. **WebSocketManager**: Real-time order and account updates via Binance WebSocket
5. **BinanceClient**: API interaction layer for order placement
6. **TradingLogger**: Comprehensive logging system creating CSV transaction logs

### Trading Strategy Flow

1. Bot calculates random price offset within market queue
2. Places POST-ONLY limit order at calculated price
3. WebSocket monitors order status continuously
4. On order fill, immediately places closing limit order at entry price + take_profit
5. Process repeats based on wait_time and max_orders configuration

## Environment Configuration

Required environment variables in `.env`:
- `API_KEY`: Binance API key with futures trading permissions
- `API_SECRET`: Binance API secret
- `TELEGRAM_BOT_TOKEN`: Telegram bot token (optional)
- `TELEGRAM_CHAT_ID`: Telegram chat ID (optional)
- `TIMEZONE`: Timezone for logging (e.g., America/New_York)

## Logging

The bot creates symbol-specific log files:
- `{SYMBOL}_transactions_log.csv`: Detailed trade data
- `{SYMBOL}_bot_activity.log`: Debug and activity logs

## Dependencies

Key packages:
- `binance-futures-connector==4.1.0`: Binance API client
- `rich==13.7.0`: Dashboard UI components
- `python-telegram-bot==20.7`: Telegram notifications
- `pandas==2.3.1`: Data analysis for transaction logs
- `python-dotenv==1.0.0`: Environment variable management

## Security Notes

- API credentials are managed via environment variables
- Never commit `.env` files (already in .gitignore)
- All connections use secure protocols (WSS/HTTPS)
- API keys should have minimal required permissions (futures trading only)