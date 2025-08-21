"""
Configuration Management System

This module provides centralized configuration management with environment variable support.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Exchange configuration"""
    exchange_id: str = 'binance'
    api_key: str = ''
    api_secret: str = ''
    testnet: bool = True
    sandbox_mode: bool = True
    timeout: int = 30000
    enable_rate_limit: bool = True

@dataclass
class TradingConfig:
    """Trading configuration"""
    symbol: str = 'BTCUSDT'
    mode: str = 'simulation'  # live, backtest, simulation
    max_position_size: float = 0.1
    min_trade_amount: float = 0.001
    max_daily_trades: int = 50
    cooldown_period: int = 60
    risk_management: bool = True
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.01

@dataclass
class BacktestConfig:
    """Backtest configuration"""
    data_file: str = ''
    initial_balance: float = 10000.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframe: str = '1m'
    include_fees: bool = True
    slippage: float = 0.001

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str = 'default'
    type: str = 'moving_average'  # moving_average, mean_reversion, grid
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_path: str = 'trading_bot.log'
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    console_output: bool = True

@dataclass
class NotificationConfig:
    """Notification configuration"""
    telegram_enabled: bool = False
    telegram_bot_token: str = ''
    telegram_chat_id: str = ''
    email_enabled: bool = False
    email_smtp_server: str = ''
    email_smtp_port: int = 587
    email_username: str = ''
    email_password: str = ''
    email_recipients: List[str] = field(default_factory=list)

@dataclass
class AppConfig:
    """Main application configuration"""
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Add default strategies if none provided
        if not self.strategies:
            self.strategies = [
                StrategyConfig(
                    name='ma_crossover',
                    type='moving_average',
                    enabled=True,
                    parameters={'fast_period': 10, 'slow_period': 30}
                ),
                StrategyConfig(
                    name='mean_reversion',
                    type='mean_reversion',
                    enabled=False,
                    parameters={'period': 20, 'std_dev_threshold': 2.0}
                ),
                StrategyConfig(
                    name='grid_trading',
                    type='grid',
                    enabled=False,
                    parameters={'grid_spacing': 0.01, 'grid_levels': 10}
                )
            ]

class ConfigManager:
    """Configuration manager"""
    
    def __init__(self, config_file: str = 'config.json'):
        self.config_file = config_file
        self.config: Optional[AppConfig] = None
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file and environment variables"""
        try:
            # Load from file if it exists
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self.config = AppConfig(**config_data)
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                # Create default configuration
                self.config = AppConfig()
                logger.info("Using default configuration")
            
            # Override with environment variables
            self._load_from_env()
            
            # Validate configuration
            self._validate_config()
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config = AppConfig()
    
    def _load_from_env(self) -> None:
        """Load configuration from environment variables"""
        # Exchange configuration
        if os.getenv('EXCHANGE_ID'):
            self.config.exchange.exchange_id = os.getenv('EXCHANGE_ID')
        if os.getenv('API_KEY'):
            self.config.exchange.api_key = os.getenv('API_KEY')
        if os.getenv('API_SECRET'):
            self.config.exchange.api_secret = os.getenv('API_SECRET')
        if os.getenv('TESTNET'):
            self.config.exchange.testnet = os.getenv('TESTNET').lower() == 'true'
        
        # Trading configuration
        if os.getenv('TRADING_SYMBOL'):
            self.config.trading.symbol = os.getenv('TRADING_SYMBOL')
        if os.getenv('TRADING_MODE'):
            self.config.trading.mode = os.getenv('TRADING_MODE')
        if os.getenv('MAX_POSITION_SIZE'):
            self.config.trading.max_position_size = float(os.getenv('MAX_POSITION_SIZE'))
        
        # Backtest configuration
        if os.getenv('BACKTEST_DATA_FILE'):
            self.config.backtest.data_file = os.getenv('BACKTEST_DATA_FILE')
        if os.getenv('INITIAL_BALANCE'):
            self.config.backtest.initial_balance = float(os.getenv('INITIAL_BALANCE'))
        
        # Logging configuration
        if os.getenv('LOG_LEVEL'):
            self.config.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.config.logging.file_path = os.getenv('LOG_FILE')
        
        # Telegram configuration
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self.config.notifications.telegram_enabled = True
            self.config.notifications.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        if os.getenv('TELEGRAM_CHAT_ID'):
            self.config.notifications.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    def _validate_config(self) -> None:
        """Validate configuration"""
        errors = []
        
        # Validate exchange configuration
        if not self.config.exchange.api_key and self.config.trading.mode == 'live':
            errors.append("API key is required for live trading")
        if not self.config.exchange.api_secret and self.config.trading.mode == 'live':
            errors.append("API secret is required for live trading")
        
        # Validate trading configuration
        if self.config.trading.max_position_size <= 0:
            errors.append("Max position size must be positive")
        if self.config.trading.min_trade_amount <= 0:
            errors.append("Min trade amount must be positive")
        if not 0 < self.config.trading.stop_loss_pct < 1:
            errors.append("Stop loss percentage must be between 0 and 1")
        if not 0 < self.config.trading.take_profit_pct < 1:
            errors.append("Take profit percentage must be between 0 and 1")
        
        # Validate backtest configuration
        if self.config.trading.mode == 'backtest':
            if not self.config.backtest.data_file:
                errors.append("Data file is required for backtest mode")
            elif not os.path.exists(self.config.backtest.data_file):
                errors.append(f"Data file not found: {self.config.backtest.data_file}")
        
        # Validate strategies
        for strategy in self.config.strategies:
            if strategy.type not in ['moving_average', 'mean_reversion', 'grid']:
                errors.append(f"Invalid strategy type: {strategy.type}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def get_config(self) -> AppConfig:
        """Get configuration"""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        try:
            # Convert nested dict to dataclass
            def update_dataclass(obj, updates):
                for key, value in updates.items():
                    if hasattr(obj, key):
                        if isinstance(value, dict) and hasattr(getattr(obj, key), '__dataclass_fields__'):
                            update_dataclass(getattr(obj, key), value)
                        else:
                            setattr(obj, key, value)
            
            update_dataclass(self.config, updates)
            self._validate_config()
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            raise
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy"""
        for strategy in self.config.strategies:
            if strategy.name == strategy_name:
                return strategy
        return None
    
    def enable_strategy(self, strategy_name: str) -> bool:
        """Enable a strategy"""
        strategy = self.get_strategy_config(strategy_name)
        if strategy:
            strategy.enabled = True
            self.save_config()
            return True
        return False
    
    def disable_strategy(self, strategy_name: str) -> bool:
        """Disable a strategy"""
        strategy = self.get_strategy_config(strategy_name)
        if strategy:
            strategy.enabled = False
            self.save_config()
            return True
        return False
    
    def add_strategy(self, strategy_config: StrategyConfig) -> None:
        """Add a new strategy"""
        self.config.strategies.append(strategy_config)
        self.save_config()
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy"""
        for i, strategy in enumerate(self.config.strategies):
            if strategy.name == strategy_name:
                del self.config.strategies[i]
                self.save_config()
                return True
        return False
    
    def create_sample_config(self, file_path: str = 'config.sample.json') -> None:
        """Create a sample configuration file"""
        sample_config = AppConfig()
        
        # Add sample API keys (placeholder)
        sample_config.exchange.api_key = "YOUR_API_KEY_HERE"
        sample_config.exchange.api_secret = "YOUR_API_SECRET_HERE"
        
        # Add sample backtest data file
        sample_config.backtest.data_file = "data/sample_ohlcv.csv"
        
        # Add sample Telegram config
        sample_config.notifications.telegram_bot_token = "YOUR_TELEGRAM_BOT_TOKEN"
        sample_config.notifications.telegram_chat_id = "YOUR_TELEGRAM_CHAT_ID"
        
        try:
            with open(file_path, 'w') as f:
                json.dump(asdict(sample_config), f, indent=2)
            logger.info(f"Sample configuration created: {file_path}")
        except Exception as e:
            logger.error(f"Error creating sample configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self.config)
    
    def get_trading_engine_config(self) -> Dict[str, Any]:
        """Get configuration for trading engine"""
        return {
            'exchange_id': self.config.exchange.exchange_id,
            'api_key': self.config.exchange.api_key,
            'api_secret': self.config.exchange.api_secret,
            'testnet': self.config.exchange.testnet,
            'sandbox_mode': self.config.exchange.sandbox_mode,
            'mode': self.config.trading.mode,
            'symbol': self.config.trading.symbol,
            'data_file': self.config.backtest.data_file,
            'initial_balance': self.config.backtest.initial_balance,
            'base_asset': self.config.trading.symbol[:-4] if self.config.trading.symbol.endswith('USDT') else 'BTC'
        }

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get global configuration"""
    return config_manager.get_config()

def get_trading_engine_config() -> Dict[str, Any]:
    """Get trading engine configuration"""
    return config_manager.get_trading_engine_config()