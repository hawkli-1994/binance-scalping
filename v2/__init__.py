"""
Binance Trading Bot v2

An advanced cryptocurrency trading system with CCXT integration,
multiple trading strategies, and comprehensive backtesting capabilities.
"""

__version__ = "2.0.0"
__author__ = "Trading Bot Team"
__email__ = "support@tradingbot.com"

from .config import config_manager, get_config, get_trading_engine_config
from .trading_engine import TradingEngine, TradingMode, Order, OrderSide, OrderType
from .strategies import StrategyManager, MovingAverageStrategy, MeanReversionStrategy, GridTradingStrategy

__all__ = [
    'config_manager',
    'get_config',
    'get_trading_engine_config',
    'TradingEngine',
    'TradingMode',
    'Order',
    'OrderSide',
    'OrderType',
    'StrategyManager',
    'MovingAverageStrategy',
    'MeanReversionStrategy',
    'GridTradingStrategy',
]