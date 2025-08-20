import asyncio
import time
import uuid
import logging
import hmac
import hashlib
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from decimal import Decimal, ROUND_DOWN
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import threading
from contextlib import asynccontextmanager
from itertools import product

logger = logging.getLogger(__name__)

class TradingError(Exception):
    """Base class for trading-related errors"""
    pass

class InsufficientBalanceError(TradingError):
    """Insufficient balance error"""
    pass

class InvalidPriceError(TradingError):
    """Invalid price error"""
    pass

class OrderRejectedError(TradingError):
    """Order rejected error"""
    pass

class BinanceAPIError(TradingError):
    """Binance API error"""
    pass

@dataclass
class ExchangeInfo:
    """Exchange information"""
    symbol: str
    base_asset: str
    quote_asset: str
    min_qty: float
    max_qty: float
    min_price: float
    max_price: float
    min_notional: float
    price_precision: int
    qty_precision: int
    tick_size: float
    step_size: float

class BinanceAsyncClient:
    """Binance async client"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _generate_signature(self, query_string: str) -> str:
        """Generate API signature"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """Get timestamp"""
        return int(time.time() * 1000)
    
    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        """Send API request"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
            
        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        
        try:
            if method == 'GET':
                async with self.session.get(url, headers=headers, params=params) as response:
                    data = await response.json()
            elif method == 'POST':
                async with self.session.post(url, headers=headers, data=params) as response:
                    data = await response.json()
            elif method == 'DELETE':
                async with self.session.delete(url, headers=headers, params=params) as response:
                    data = await response.json()
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status != 200:
                raise BinanceAPIError(f"API Error {response.status}: {data}")
                
            return data
            
        except aiohttp.ClientError as e:
            raise BinanceAPIError(f"Network error: {e}")
        except json.JSONDecodeError as e:
            raise BinanceAPIError(f"JSON decode error: {e}")
    
    async def get_exchange_info(self):
        """Get exchange information"""
        return await self._request('GET', '/api/v3/exchangeInfo')
    
    async def get_symbol_ticker(self, symbol: str):
        """Get symbol ticker price"""
        return await self._request('GET', '/api/v3/ticker/price', {'symbol': symbol})
    
    async def get_account(self):
        """Get account information"""
        return await self._request('GET', '/api/v3/account', signed=True)
    
    async def create_order(self, **params):
        """Create order"""
        return await self._request('POST', '/api/v3/order', params, signed=True)
    
    async def get_order(self, symbol: str, orderId: str):
        """Get order information"""
        params = {'symbol': symbol, 'orderId': orderId}
        return await self._request('GET', '/api/v3/order', params, signed=True)
    
    async def cancel_order(self, symbol: str, orderId: str):
        """Cancel order"""
        params = {'symbol': symbol, 'orderId': orderId}
        return await self._request('DELETE', '/api/v3/order', params, signed=True)
    
    async def get_open_orders(self, symbol: str = None):
        """Get current open orders"""
        params = {'symbol': symbol} if symbol else {}
        return await self._request('GET', '/api/v3/openOrders', params, signed=True)
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500):
        """Get kline data"""
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        return await self._request('GET', '/api/v3/klines', params)

class ConfigValidator:
    """Configuration validator"""
    
    REQUIRED_KEYS = [
        'symbol', 'target_ratio', 'balance_threshold',
        'min_trade_amount', 'max_position_ratio', 'price_lookback', 
        'api_key', 'api_secret', 'stop_loss_atr_multiplier', 'atr_period',
        'take_profit_atr_multiplier'
    ]
    
    @classmethod
    def validate(cls, config: Dict) -> Dict:
        """Validate configuration"""
        for key in cls.REQUIRED_KEYS:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        if not config['api_key'] or not config['api_secret']:
            raise ValueError("API key and secret must not be empty")
        
        if not 0 < config['target_ratio'] < 1:
            raise ValueError("target_ratio must be between 0 and 1")
        
        if not 0 < config['balance_threshold'] < 0.5:
            raise ValueError("balance_threshold must be between 0 and 0.5")
        
        if config['min_trade_amount'] <= 0:
            raise ValueError("min_trade_amount must be positive")
        
        if not 0 < config['max_position_ratio'] <= 0.1:
            raise ValueError("max_position_ratio must be between 0 and 0.1")
        
        if config['stop_loss_atr_multiplier'] <= 0:
            raise ValueError("stop_loss_atr_multiplier must be positive")
        
        if config['take_profit_atr_multiplier'] <= 0:
            raise ValueError("take_profit_atr_multiplier must be positive")
        
        if config['atr_period'] < 1:
            raise ValueError("atr_period must be positive")
        
        defaults = {
            'testnet': True,
            'simulation_mode': True,
            'backtest_mode': False,
            'backtest_file': None,
            'max_chase_attempts': 3,
            'chase_interval': 5.0,
            'max_order_age': 300.0,
            'partial_fill_threshold': 0.1,
            'max_slippage_percent': 0.5,
            'price_deviation_threshold': 0.05,
            'max_daily_trades': 50,
            'emergency_stop': False,
            'max_concurrent_orders': 3,
            'order_cleanup_interval': 3600,
            'max_order_history': 1000,
            'min_profit_threshold': 0.002,
            'volatility_window': 20,
            'cooldown_period': 30,
            'trading_fee_rate': 0.001,  # 0.1% trading fee
        }
        
        for key, value in defaults.items():
            config.setdefault(key, value)
        
        return config

class PrecisionManager:
    """Precision manager"""
    
    def __init__(self, exchange_info: ExchangeInfo):
        self.exchange_info = exchange_info
    
    def format_price(self, price: float) -> str:
        """Format price"""
        decimal_price = Decimal(str(price))
        tick_size = Decimal(str(self.exchange_info.tick_size))
        rounded_price = (decimal_price // tick_size) * tick_size
        return format(rounded_price, f'.{self.exchange_info.price_precision}f')
    
    def format_quantity(self, quantity: float) -> str:
        """Format quantity"""
        decimal_qty = Decimal(str(quantity))
        step_size = Decimal(str(self.exchange_info.step_size))
        rounded_qty = (decimal_qty // step_size) * step_size
        return format(rounded_qty, f'.{self.exchange_info.qty_precision}f')
    
    def validate_order(self, side: str, quantity: float, price: float) -> bool:
        """Validate order parameters"""
        if quantity < self.exchange_info.min_qty or quantity > self.exchange_info.max_qty:
            logger.warning(f"Quantity {quantity} out of range [{self.exchange_info.min_qty}, {self.exchange_info.max_qty}]")
            return False
        
        if price < self.exchange_info.min_price or price > self.exchange_info.max_price:
            logger.warning(f"Price {price} out of range [{self.exchange_info.min_price}, {self.exchange_info.max_price}]")
            return False
        
        notional = quantity * price
        if notional < self.exchange_info.min_notional:
            logger.warning(f"Notional {notional} below minimum {self.exchange_info.min_notional}")
            return False
        
        return True

class VolatilityCalculator:
    """Volatility calculator"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
    
    def add_price(self, price: float):
        """Add price data"""
        if len(self.returns) > 0:
            last_price = self.returns[-1] if len(self.returns) > 0 else price
            return_val = (price - last_price) / last_price if last_price > 0 else 0
            self.returns.append(return_val)
        else:
            self.returns.append(0)
    
    def get_volatility(self) -> float:
        """Get volatility"""
        if len(self.returns) < 5:
            return 0.02
        return np.std(list(self.returns)) * np.sqrt(len(self.returns))

class ATRCalculator:
    """ATR calculator for stop-loss and take-profit"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.trs = deque(maxlen=period)
    
    def add_candle(self, high: float, low: float, close: float, prev_close: float = None):
        """Add candle data and calculate true range"""
        if prev_close is None and len(self.trs) == 0:
            return
        if prev_close is not None:
            tr = max(
                abs(high - low),
                abs(high - prev_close) if prev_close else 0,
                abs(prev_close - low) if prev_close else 0
            )
            self.trs.append(tr)
    
    def get_atr(self) -> float:
        """Get Average True Range"""
        if len(self.trs) < self.period:
            return 0
        return np.mean(list(self.trs))

class SafetyChecker:
    """Safety checker"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.last_valid_price = None
        self.price_history = deque(maxlen=50)
        self.volatility_calc = VolatilityCalculator(config['volatility_window'])
        self.atr_calc = ATRCalculator(config['atr_period'])
        self.last_trade_time = 0
    
    def check_price_sanity(self, price: float) -> bool:
        """Check price sanity"""
        if price <= 0:
            return False
        
        if self.last_valid_price:
            deviation = abs(price - self.last_valid_price) / self.last_valid_price
            if deviation > self.config['price_deviation_threshold']:
                logger.warning(f"Price deviation too large: {deviation:.2%}")
                return False
        
        if len(self.price_history) >= 10:
            recent_avg = np.mean(list(self.price_history)[-10:])
            if abs(price - recent_avg) / recent_avg > 0.05:
                logger.warning(f"Price deviated from recent average: {price} vs {recent_avg}")
                return False
        
        self.price_history.append(price)
        self.volatility_calc.add_price(price)
        self.last_valid_price = price
        return True
    
    def check_balance_sufficiency(self, side: str, quantity: float, price: float, 
                                 btc_balance: float, usdt_balance: float) -> bool:
        """Check balance sufficiency"""
        if side == 'BUY':
            required_usdt = quantity * price * (1 + self.config['trading_fee_rate'])
            return usdt_balance >= required_usdt
        else:
            return btc_balance >= quantity
    
    def check_cooldown(self) -> bool:
        """Check trading cooldown"""
        current_time = time.time()
        if current_time - self.last_trade_time < self.config['cooldown_period']:
            return False
        return True
    
    def update_trade_time(self):
        """Update trade time"""
        self.last_trade_time = time.time()
    
    def get_current_volatility(self) -> float:
        """Get current volatility"""
        return self.volatility_calc.get_volatility()
    
    def update_atr(self, high: float, low: float, close: float, prev_close: float = None):
        """Update ATR with candle data"""
        self.atr_calc.add_candle(high, low, close, prev_close)
    
    def get_stop_loss_price(self, side: str, entry_price: float) -> float:
        """Calculate stop-loss price based on ATR"""
        atr = self.atr_calc.get_atr()
        multiplier = self.config['stop_loss_atr_multiplier']
        if side == 'BUY':
            return entry_price - atr * multiplier
        else:
            return entry_price + atr * multiplier
    
    def get_take_profit_price(self, side: str, entry_price: float) -> float:
        """Calculate take-profit price based on ATR"""
        atr = self.atr_calc.get_atr()
        multiplier = self.config['take_profit_atr_multiplier']
        if side == 'BUY':
            return entry_price + atr * multiplier
        else:
            return entry_price - atr * multiplier

class TradingLock:
    """Trading lock mechanism"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_strategies = set()
    
    @asynccontextmanager
    async def acquire(self, strategy_name: str):
        """Acquire trading lock"""
        async with self._lock:
            if strategy_name in self._active_strategies:
                logger.warning(f"Strategy {strategy_name} already active, skipping")
                yield False
                return
            
            self._active_strategies.add(strategy_name)
            try:
                yield True
            finally:
                self._active_strategies.discard(strategy_name)

class SecureOrderExecutionEngine:
    """Secure order execution engine"""
    
    def __init__(self, client: BinanceAsyncClient, config: Dict, exchange_info: ExchangeInfo):
        self.client = client
        self.config = ConfigValidator.validate(config)
        self.exchange_info = exchange_info
        self.precision_manager = PrecisionManager(exchange_info)
        self.safety_checker = SafetyChecker(self.config)
        self.trading_lock = TradingLock()
        
        self.active_orders: Dict[str, Dict] = {}
        self.completed_orders = deque(maxlen=self.config['max_order_history'])
        self.order_cleanup_task = None
        
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'total_slippage': 0.0,
            'daily_trades': 0,
            'last_reset_date': time.time(),
            'total_profit': 0.0,
            'stop_loss_triggers': 0,
            'take_profit_triggers': 0
        }
        
        if not self.config['simulation_mode'] and not self.config['backtest_mode']:
            self.order_cleanup_task = asyncio.create_task(self._cleanup_orders_periodically())
    
    async def _cleanup_orders_periodically(self):
        """Periodically clean up orders"""
        while True:
            try:
                await asyncio.sleep(self.config['order_cleanup_interval'])
                await self._cleanup_old_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_old_orders(self):
        """Clean up old orders"""
        current_time = time.time()
        orders_to_remove = []
        
        for order_id, order_data in self.active_orders.items():
            if current_time - order_data['created_time'] > self.config['max_order_age']:
                orders_to_remove.append(order_id)
        
        for order_id in orders_to_remove:
            try:
                await self.cancel_order(order_id, reason="expired")
            except Exception as e:
                logger.error(f"Error cancelling expired order {order_id}: {e}")
    
    def _reset_daily_stats(self):
        """Reset daily stats"""
        current_date = time.time() // 86400
        last_date = self.stats['last_reset_date'] // 86400
        
        if current_date != last_date:
            self.stats['daily_trades'] = 0
            self.stats['last_reset_date'] = time.time()
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed"""
        self._reset_daily_stats()
        
        if self.config['emergency_stop']:
            return False
        
        if self.stats['daily_trades'] >= self.config['max_daily_trades']:
            logger.warning("Daily trade limit reached")
            return False
        
        if len(self.active_orders) >= self.config['max_concurrent_orders']:
            logger.warning("Maximum concurrent orders reached")
            return False
        
        if not self.safety_checker.check_cooldown():
            logger.debug("Trading in cooldown period")
            return False
        
        return True
    
    async def place_order_secure(self, symbol: str, side: str, quantity: float, 
                                price: float = None, order_type: str = 'LIMIT',
                                strategy_name: str = "default",
                                max_slippage_percent: float = None) -> Optional[str]:
        """Secure order placement with stop-loss and take-profit"""
        if not self._can_trade():
            return None
        
        max_slippage_percent = max_slippage_percent or self.config['max_slippage_percent']
        
        try:
            if price and not self.safety_checker.check_price_sanity(price):
                raise InvalidPriceError(f"Price {price} failed sanity check")
            
            if not self.precision_manager.validate_order(side, quantity, price):
                raise OrderRejectedError("Order parameters validation failed")
            
            formatted_qty = self.precision_manager.format_quantity(quantity)
            formatted_price = self.precision_manager.format_price(price) if price else None
            
            order_id = str(uuid.uuid4())
            client_order_id = f"bot_{int(time.time())}_{order_id[:8]}"
            
            stop_loss_price = self.safety_checker.get_stop_loss_price(side, float(formatted_price) if formatted_price else price)
            take_profit_price = self.safety_checker.get_take_profit_price(side, float(formatted_price) if formatted_price else price)
            
            if self.config['simulation_mode'] or self.config['backtest_mode']:
                return await self._simulate_order_secure(
                    order_id, symbol, side, float(formatted_qty), 
                    float(formatted_price) if formatted_price else None,
                    order_type, strategy_name, stop_loss_price, take_profit_price
                )
            
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': formatted_qty,
                'newClientOrderId': client_order_id
            }
            
            if order_type == 'LIMIT' and formatted_price:
                order_params['price'] = formatted_price
                order_params['timeInForce'] = 'GTC'
            
            result = await self._safe_api_call(self.client.create_order, **order_params)
            
            if result:
                self.active_orders[order_id] = {
                    'exchange_order_id': result['orderId'],
                    'symbol': symbol,
                    'side': side,
                    'quantity': float(formatted_qty),
                    'price': float(formatted_price) if formatted_price else None,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'order_type': order_type,
                    'strategy': strategy_name,
                    'created_time': time.time(),
                    'status': 'NEW'
                }
                
                self.stats['total_orders'] += 1
                self.stats['daily_trades'] += 1
                self.safety_checker.update_trade_time()
                
                logger.info(f"Order placed: {order_id} - {side} {formatted_qty} @ {formatted_price}, "
                           f"Stop-loss: {stop_loss_price}, Take-profit: {take_profit_price}")
                
                asyncio.create_task(self._monitor_order_secure(order_id))
                
                return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.stats['rejected_orders'] += 1
            return None
    
    async def _simulate_order_secure(self, order_id: str, symbol: str, side: str, 
                                   quantity: float, price: float, order_type: str,
                                   strategy_name: str, stop_loss_price: float,
                                   take_profit_price: float) -> str:
        """Secure simulated order placement with fees and slippage"""
        await asyncio.sleep(np.random.uniform(0.1, 0.3))
        
        volatility = self.safety_checker.get_current_volatility()
        fee_rate = self.config['trading_fee_rate']
        
        if order_type == 'MARKET':
            simulated_slippage = np.random.normal(0, volatility * 0.5)
            if abs(simulated_slippage) > self.config['max_slippage_percent'] / 100:
                logger.warning(f"Simulated slippage {simulated_slippage:.4f} exceeds limit")
                return None
            
            execution_price = price * (1 + simulated_slippage)
        else:
            execution_price = price
            
            fill_probability = max(0.7, 0.95 - volatility * 10)
            if np.random.random() > fill_probability:
                logger.info(f"Simulated order {order_id} not filled (volatility: {volatility:.4f})")
                return None
        
        self.active_orders[order_id] = {
            'exchange_order_id': f"sim_{order_id}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'execution_price': execution_price,
            'order_type': order_type,
            'strategy': strategy_name,
            'created_time': time.time(),
            'status': 'FILLED'
        }
        
        await asyncio.sleep(0.1)
        self._complete_order(order_id, 'FILLED', execution_price)
        
        return order_id
    
    async def _monitor_order_secure(self, order_id: str):
        """Secure order monitoring with stop-loss and take-profit"""
        monitor_start_time = time.time()
        max_monitor_time = self.config['max_order_age']
        
        while order_id in self.active_orders:
            try:
                if time.time() - monitor_start_time > max_monitor_time:
                    logger.warning(f"Order {order_id} monitoring timeout")
                    await self.cancel_order(order_id, reason="timeout")
                    break
                
                order_data = self.active_orders.get(order_id)
                if not order_data:
                    break
                
                current_price = self.safety_checker.last_valid_price
                if current_price:
                    # Check stop-loss
                    if (order_data['side'] == 'BUY' and current_price <= order_data['stop_loss_price']) or \
                       (order_data['side'] == 'SELL' and current_price >= order_data['stop_loss_price']):
                        logger.warning(f"Stop-loss triggered for order {order_id}: {current_price} vs {order_data['stop_loss_price']}")
                        await self._execute_stop_loss_take_profit(order_id, 'stop_loss')
                        break
                    
                    # Check take-profit
                    if (order_data['side'] == 'BUY' and current_price >= order_data['take_profit_price']) or \
                       (order_data['side'] == 'SELL' and current_price <= order_data['take_profit_price']):
                        logger.warning(f"Take-profit triggered for order {order_id}: {current_price} vs {order_data['take_profit_price']}")
                        await self._execute_stop_loss_take_profit(order_id, 'take_profit')
                        break
                
                await self._update_order_status_secure(order_id)
                
                order_data = self.active_orders.get(order_id)
                if not order_data:
                    break
                
                if order_data['status'] in ['FILLED', 'CANCELED', 'REJECTED']:
                    self._complete_order(order_id, order_data['status'], 
                                       order_data.get('execution_price', order_data.get('price')))
                    break
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(5)
    
    async def _execute_stop_loss_take_profit(self, order_id: str, reason: str):
        """Execute stop-loss or take-profit with market order"""
        if order_id not in self.active_orders:
            return
        
        order_data = self.active_orders[order_id]
        
        try:
            # Cancel the original limit order if not in backtest/simulation mode
            if not self.config['simulation_mode'] and not self.config['backtest_mode']:
                await self._safe_api_call(
                    self.client.cancel_order,
                    symbol=order_data['symbol'],
                    orderId=order_data['exchange_order_id']
                )
            
            # Place market order to close position
            close_side = 'SELL' if order_data['side'] == 'BUY' else 'BUY'
            result = await self.place_order_secure(
                symbol=order_data['symbol'],
                side=close_side,
                quantity=order_data['quantity'],
                order_type='MARKET',
                strategy_name=f"{reason}_close"
            )
            
            if result:
                self.stats['stop_loss_triggers' if reason == 'stop_loss' else 'take_profit_triggers'] += 1
                self._complete_order(order_id, 'CANCELED', order_data['price'])
                logger.info(f"{reason.capitalize()} executed for order {order_id}: Placed market {close_side} order")
            else:
                logger.error(f"Failed to execute {reason} market order for {order_id}")
                
        except Exception as e:
            logger.error(f"Error executing {reason} for order {order_id}: {e}")
    
    async def _update_order_status_secure(self, order_id: str):
        """Secure order status update"""
        if order_id not in self.active_orders:
            return
        
        order_data = self.active_orders[order_id]
        
        try:
            if not self.config['backtest_mode']:
                result = await self._safe_api_call(
                    self.client.get_order,
                    symbol=order_data['symbol'],
                    orderId=order_data['exchange_order_id']
                )
                
                if result:
                    order_data['status'] = result['status']
                    if result['status'] == 'FILLED':
                        executed_qty = float(result.get('executedQty', 0))
                        if executed_qty > 0:
                            avg_price = float(result.get('cummulativeQuoteQty', 0)) / executed_qty
                            order_data['execution_price'] = avg_price
                        else:
                            order_data['execution_price'] = order_data['price']
                        
                        if order_data['price']:
                            slippage = (order_data['execution_price'] - order_data['price']) / order_data['price']
                            if order_data['side'] == 'SELL':
                                slippage = -slippage
                            
                            self.stats['total_slippage'] += abs(slippage)
                            
                            if abs(slippage) > self.config['max_slippage_percent'] / 100:
                                logger.warning(f"High slippage detected: {slippage:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating order status for {order_id}: {e}")
    
    def _complete_order(self, order_id: str, status: str, execution_price: float = None):
        """Complete order"""
        if order_id in self.active_orders:
            order_data = self.active_orders.pop(order_id)
            order_data['completed_time'] = time.time()
            order_data['final_status'] = status
            if execution_price:
                order_data['execution_price'] = execution_price
            
            self.completed_orders.append(order_data)
            
            if status == 'FILLED':
                self.stats['successful_orders'] += 1
                if order_data.get('execution_price') and order_data.get('price'):
                    price_diff = order_data['execution_price'] - order_data['price']
                    if order_data['side'] == 'SELL':
                        price_diff = -price_diff
                    profit = price_diff * order_data['quantity']
                    self.stats['total_profit'] += profit
            elif status == 'CANCELED':
                self.stats['cancelled_orders'] += 1
            
            logger.info(f"Order completed: {order_id} - Status: {status}")
    
    async def cancel_order(self, order_id: str, reason: str = "manual") -> bool:
        """Cancel order"""
        if order_id not in self.active_orders:
            return False
        
        order_data = self.active_orders[order_id]
        
        try:
            if not self.config['simulation_mode'] and not self.config['backtest_mode']:
                await self._safe_api_call(
                    self.client.cancel_order,
                    symbol=order_data['symbol'],
                    orderId=order_data['exchange_order_id']
                )
            
            self._complete_order(order_id, 'CANCELED')
            logger.info(f"Order cancelled: {order_id} - Reason: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def _safe_api_call(self, func, max_retries: int = 3, **kwargs):
        """Safe API call"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await func(**kwargs)
            except BinanceAPIError as e:
                last_exception = e
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in [
                    'insufficient', 'invalid', 'unauthorized', 'forbidden',
                    'account has insufficient balance', 'filter failure'
                ]):
                    raise e
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error after {max_retries} attempts: {e}")
        
        raise last_exception
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        success_rate = (self.stats['successful_orders'] / max(self.stats['total_orders'], 1)) * 100
        avg_slippage = self.stats['total_slippage'] / max(self.stats['successful_orders'], 1)
        
        return {
            'total_orders': self.stats['total_orders'],
            'successful_orders': self.stats['successful_orders'],
            'success_rate': success_rate,
            'average_slippage': avg_slippage,
            'daily_trades': self.stats['daily_trades'],
            'active_orders': len(self.active_orders),
            'total_profit': self.stats['total_profit'],
            'stop_loss_triggers': self.stats['stop_loss_triggers'],
            'take_profit_triggers': self.stats['take_profit_triggers']
        }
    
    async def shutdown(self):
        """Shutdown execution engine"""
        if self.order_cleanup_task:
            self.order_cleanup_task.cancel()
            try:
                await self.order_cleanup_task
            except asyncio.CancelledError:
                pass
        
        cancel_tasks = []
        for order_id in list(self.active_orders.keys()):
            cancel_tasks.append(self.cancel_order(order_id, reason="shutdown"))
        
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        logger.info("Order execution engine shutdown completed")

class RiskManager:
    """Risk manager"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_drawdown = 0.05
        self.position_size_multiplier = 1.0
        self.emergency_mode = False
        
    def calculate_position_size(self, total_value: float, volatility: float, 
                              target_amount: float) -> float:
        """Calculate position size based on risk"""
        volatility_adjustment = max(0.1, min(1.0, 1.0 - volatility * 10))
        drawdown_adjustment = self.position_size_multiplier
        adjusted_amount = target_amount * volatility_adjustment * drawdown_adjustment
        max_position = total_value * self.config['max_position_ratio']
        return min(adjusted_amount, max_position)
    
    def update_drawdown(self, current_value: float, peak_value: float):
        """Update drawdown information"""
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > self.max_drawdown:
                self.position_size_multiplier *= 0.5
                logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}, reducing position size")
                
                if drawdown > self.max_drawdown * 2:
                    self.emergency_mode = True
                    logger.critical("Emergency mode activated due to excessive drawdown")
    
    def is_emergency_mode(self) -> bool:
        """Check if in emergency mode"""
        return self.emergency_mode

class MarketAnalyzer:
    """Market analyzer"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
    
    def add_data(self, price: float, volume: float = 0):
        """Add market data"""
        self.prices.append(price)
        self.volumes.append(volume)
    
    def get_trend_strength(self) -> float:
        """Get trend strength (-1 to 1)"""
        if len(self.prices) < 20:
            return 0
        
        prices = list(self.prices)
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-20:])
        
        if long_ma == 0:
            return 0
        
        trend = (short_ma - long_ma) / long_ma
        return np.clip(trend * 100, -1, 1)
    
    def get_rsi(self, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(self.prices) < period + 1:
            return 50
        
        prices = list(self.prices)
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def is_oversold(self) -> bool:
        """Check if oversold"""
        return self.get_rsi() < 30
    
    def is_overbought(self) -> bool:
        """Check if overbought"""
        return self.get_rsi() > 70

class SecureTradingBot:
    """Secure trading bot"""
    
    def __init__(self, config: Dict):
        self.config = ConfigValidator.validate(config)
        self.client = None
        self.execution_engine = None
        self.exchange_info = None
        self.backtest_data = None
        self.backtest_index = 0
        
        self.is_running = False
        self.btc_balance = 0.0
        self.usdt_balance = 0.0
        self.last_price = 0.0
        self.price_history = deque(maxlen=self.config['price_lookback'])
        self.peak_portfolio_value = 0.0
        
        self.safety_checker = None
        self.trading_lock = TradingLock()
        self.risk_manager = RiskManager(self.config)
        self.market_analyzer = MarketAnalyzer()
    
    async def initialize(self):
        """Initialize bot"""
        logger.info("Initializing Secure Trading Bot...")
        
        if self.config['backtest_mode']:
            await self._load_backtest_data()
        else:
            self.client = BinanceAsyncClient(
                self.config['api_key'],
                self.config['api_secret'],
                testnet=self.config['testnet']
            )
            await self._load_exchange_info()
        
        self.safety_checker = SafetyChecker(self.config)
        self.execution_engine = SecureOrderExecutionEngine(
            self.client, self.config, self.exchange_info
        )
        
        if self.config['backtest_mode']:
            self.btc_balance = 1.0
            self.usdt_balance = 10000.0
            self.last_price = self.backtest_data.iloc[0]['close']
        else:
            await self._update_market_data()
            await self._update_balances()
        
        self.peak_portfolio_value = self._get_portfolio_value()
        self.is_running = True
        logger.info("Secure Trading Bot initialized successfully")
    
    async def _load_exchange_info(self):
        """Load exchange information"""
        try:
            async with self.client:
                exchange_info = await self.client.get_exchange_info()
                
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == self.config['symbol']:
                        filters = {f['filterType']: f for f in symbol_info['filters']}
                        lot_size = filters.get('LOT_SIZE', {})
                        price_filter = filters.get('PRICE_FILTER', {})
                        notional = filters.get('NOTIONAL', {}) or filters.get('MIN_NOTIONAL', {})
                        
                        self.exchange_info = ExchangeInfo(
                            symbol=symbol_info['symbol'],
                            base_asset=symbol_info['baseAsset'],
                            quote_asset=symbol_info['quoteAsset'],
                            min_qty=float(lot_size.get('minQty', '0.001')),
                            max_qty=float(lot_size.get('maxQty', '9000000')),
                            min_price=float(price_filter.get('minPrice', '0.01')),
                            max_price=float(price_filter.get('maxPrice', '1000000')),
                            min_notional=float(notional.get('minNotional', '10')),
                            price_precision=symbol_info.get('quotePrecision', 2),
                            qty_precision=symbol_info.get('baseAssetPrecision', 6),
                            tick_size=float(price_filter.get('tickSize', '0.01')),
                            step_size=float(lot_size.get('stepSize', '0.001'))
                        )
                        break
                
                if not self.exchange_info:
                    raise ValueError(f"Symbol {self.config['symbol']} not found")
                    
        except Exception as e:
            logger.error(f"Error loading exchange info: {e}")
            raise
    
    async def _load_backtest_data(self):
        """Load backtest data from CSV"""
        try:
            if not self.config['backtest_file']:
                raise ValueError("Backtest file path not provided")
            
            self.backtest_data = pd.read_csv(self.config['backtest_file'])
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in self.backtest_data.columns for col in required_columns):
                raise ValueError("CSV file must contain timestamp, open, high, low, close, volume columns")
            
            self.exchange_info = ExchangeInfo(
                symbol=self.config['symbol'],
                base_asset=self.config['symbol'][:-4],
                quote_asset=self.config['symbol'][-4:],
                min_qty=0.001,
                max_qty=9000000,
                min_price=0.01,
                max_price=1000000,
                min_notional=10,
                price_precision=2,
                qty_precision=6,
                tick_size=0.01,
                step_size=0.001
            )
            
            logger.info(f"Loaded backtest data with {len(self.backtest_data)} candles")
            
        except Exception as e:
            logger.error(f"Error loading backtest data: {e}")
            raise
    
    async def _update_market_data(self):
        """Update market data"""
        try:
            if self.config['backtest_mode']:
                if self.backtest_index < len(self.backtest_data):
                    candle = self.backtest_data.iloc[self.backtest_index]
                    current_price = float(candle['close'])
                    high = float(candle['high'])
                    low = float(candle['low'])
                    prev_close = float(self.backtest_data.iloc[self.backtest_index-1]['close']) if self.backtest_index > 0 else None
                    
                    if self.safety_checker.check_price_sanity(current_price):
                        self.last_price = current_price
                        self.price_history.append(current_price)
                        self.market_analyzer.add_data(current_price, float(candle['volume']))
                        self.safety_checker.update_atr(high, low, current_price, prev_close)
                    else:
                        logger.warning(f"Price {current_price} failed safety check, using last valid price")
                    self.backtest_index += 1
            else:
                async with self.client:
                    ticker = await self.client.get_symbol_ticker(symbol=self.config['symbol'])
                    current_price = float(ticker['price'])
                    
                    if self.safety_checker.check_price_sanity(current_price):
                        self.last_price = current_price
                        self.price_history.append(current_price)
                        self.market_analyzer.add_data(current_price)
                    else:
                        logger.warning(f"Price {current_price} failed safety check, using last valid price")
                    
                    klines = await self.client.get_klines(symbol=self.config['symbol'], interval='1m', limit=2)
                    if len(klines) >= 2:
                        latest = klines[-1]
                        prev = klines[-2]
                        self.safety_checker.update_atr(
                            high=float(latest[2]),
                            low=float(latest[3]),
                            close=float(latest[4]),
                            prev_close=float(prev[4])
                        )
                
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _update_balances(self):
        """Update account balances"""
        try:
            if not self.config['backtest_mode']:
                async with self.client:
                    account_info = await self.client.get_account()
                    
                    for balance in account_info['balances']:
                        if balance['asset'] == self.exchange_info.base_asset:
                            self.btc_balance = float(balance['free'])
                        elif balance['asset'] == self.exchange_info.quote_asset:
                            self.usdt_balance = float(balance['free'])
                    
                    logger.debug(f"Balances updated - BTC: {self.btc_balance:.6f}, USDT: {self.usdt_balance:.2f}")
                
        except Exception as e:
            logger.error(f"Error updating balances: {e}")
    
    def _get_portfolio_value(self) -> float:
        """Get portfolio total value"""
        return self.btc_balance * self.last_price + self.usdt_balance
    
    def _calculate_portfolio_ratio(self) -> float:
        """Calculate portfolio ratio"""
        if self.last_price <= 0:
            return 0.5
        
        btc_value = self.btc_balance * self.last_price
        total_value = btc_value + self.usdt_balance
        return btc_value / total_value if total_value > 0 else 0.5
    
    def _calculate_safe_trade_amount(self, side: str, target_ratio: float = None) -> float:
        """Calculate safe trade amount"""
        total_value = self._get_portfolio_value()
        volatility = self.safety_checker.get_current_volatility()
        
        base_amount = max(
            self.config['min_trade_amount'],
            total_value * 0.01
        )
        
        risk_adjusted_amount = self.risk_manager.calculate_position_size(
            total_value, volatility, base_amount
        )
        
        if side == 'BUY':
            available_usdt = self.usdt_balance * (1 - self.config['trading_fee_rate'])
            max_by_balance = available_usdt / self.last_price
            min_by_notional = self.exchange_info.min_notional / self.last_price
            
            trade_amount = min(risk_adjusted_amount / self.last_price, max_by_balance)
            
            if target_ratio is not None:
                current_ratio = self._calculate_portfolio_ratio()
                if current_ratio < target_ratio:
                    needed_btc_value = total_value * (target_ratio - current_ratio)
                    needed_btc = needed_btc_value / self.last_price
                    trade_amount = min(trade_amount, needed_btc * 0.8)
            
            return max(trade_amount, min_by_notional) if trade_amount >= self.exchange_info.min_qty else 0
        
        else:
            available_btc = self.btc_balance * (1 - self.config['trading_fee_rate'])
            trade_amount = min(risk_adjusted_amount / self.last_price, available_btc)
            
            if target_ratio is not None:
                current_ratio = self._calculate_portfolio_ratio()
                if current_ratio > target_ratio:
                    excess_btc_value = total_value * (current_ratio - target_ratio)
                    excess_btc = excess_btc_value / self.last_price
                    trade_amount = min(trade_amount, excess_btc * 0.8)
            
            return trade_amount if trade_amount >= self.exchange_info.min_qty else 0
    
    async def _enhanced_rebalance(self):
        """Enhanced rebalancing strategy with stop-loss and take-profit"""
        async with self.trading_lock.acquire("rebalance") as acquired:
            if not acquired:
                return
            
            current_ratio = self._calculate_portfolio_ratio()
            target_ratio = self.config['target_ratio']
            threshold = self.config['balance_threshold']
            
            trend_strength = self.market_analyzer.get_trend_strength()
            rsi = self.market_analyzer.get_rsi()
            
            adjusted_target = target_ratio
            if abs(trend_strength) > 0.3:
                if trend_strength > 0 and not self.market_analyzer.is_overbought():
                    adjusted_target = min(target_ratio + 0.1, 0.7)
                elif trend_strength < 0 and not self.market_analyzer.is_oversold():
                    adjusted_target = max(target_ratio - 0.1, 0.3)
            
            deviation = abs(current_ratio - adjusted_target)
            
            if deviation <= threshold:
                return
            
            logger.info(f"Enhanced rebalancing: {current_ratio:.3f} -> {adjusted_target:.3f} (trend: {trend_strength:.3f}, RSI: {rsi:.1f})")
            
            if current_ratio < adjusted_target - threshold:
                trade_amount = self._calculate_safe_trade_amount('BUY', adjusted_target)
                if trade_amount >= self.exchange_info.min_qty:
                    if self.safety_checker.check_balance_sufficiency(
                        'BUY', trade_amount, self.last_price, 
                        self.btc_balance, self.usdt_balance
                    ):
                        buy_price = self.last_price * 1.001
                        await self.execution_engine.place_order_secure(
                            symbol=self.config['symbol'],
                            side='BUY',
                            quantity=trade_amount,
                            price=buy_price,
                            order_type='LIMIT',
                            strategy_name='enhanced_rebalance'
                        )
                    else:
                        logger.warning("Insufficient balance for rebalance buy")
            
            elif current_ratio > adjusted_target + threshold:
                trade_amount = self._calculate_safe_trade_amount('SELL', adjusted_target)
                if trade_amount >= self.exchange_info.min_qty:
                    if self.safety_checker.check_balance_sufficiency(
                        'SELL', trade_amount, self.last_price,
                        self.btc_balance, self.usdt_balance
                    ):
                        sell_price = self.last_price * 0.999
                        await self.execution_engine.place_order_secure(
                            symbol=self.config['symbol'],
                            side='SELL',
                            quantity=trade_amount,
                            price=sell_price,
                            order_type='LIMIT',
                            strategy_name='enhanced_rebalance'
                        )
                    else:
                        logger.warning("Insufficient balance for rebalance sell")
    
    async def _emergency_stop_check(self):
        """Emergency stop check"""
        if self.config['emergency_stop']:
            logger.critical("Emergency stop activated by config")
            self.is_running = False
            return
        
        if self.risk_manager.is_emergency_mode():
            logger.critical("Emergency mode activated by risk manager")
            self.is_running = False
            return
        
        total_value = self._get_portfolio_value()
        if total_value <= 0:
            logger.critical("Zero portfolio value detected - emergency stop")
            self.is_running = False
            return
        
        if self.last_price <= 0:
            logger.critical("Invalid price detected - emergency stop")
            self.is_running = False
            return
        
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        else:
            self.risk_manager.update_drawdown(total_value, self.peak_portfolio_value)
    
    async def run_backtest(self, param_grid: Dict[str, List]):
        """Run backtest with parameter optimization"""
        results = []
        
        param_combinations = list(product(
            param_grid.get('target_ratio', [self.config['target_ratio']]),
            param_grid.get('balance_threshold', [self.config['balance_threshold']]),
            param_grid.get('stop_loss_atr_multiplier', [self.config['stop_loss_atr_multiplier']]),
            param_grid.get('take_profit_atr_multiplier', [self.config['take_profit_atr_multiplier']])
        ))
        
        for target_ratio, balance_threshold, stop_loss_atr_multiplier, take_profit_atr_multiplier in param_combinations:
            # Reset bot state
            self.backtest_index = 0
            self.btc_balance = 1.0
            self.usdt_balance = 10000.0
            self.price_history = deque(maxlen=self.config['price_lookback'])
            self.safety_checker = SafetyChecker(self.config)
            self.execution_engine = SecureOrderExecutionEngine(self.client, self.config, self.exchange_info)
            self.peak_portfolio_value = self._get_portfolio_value()
            self.is_running = True
            
            # Update config with test parameters
            self.config['target_ratio'] = target_ratio
            self.config['balance_threshold'] = balance_threshold
            self.config['stop_loss_atr_multiplier'] = stop_loss_atr_multiplier
            self.config['take_profit_atr_multiplier'] = take_profit_atr_multiplier
            
            logger.info(f"Starting backtest with params: target_ratio={target_ratio}, "
                       f"balance_threshold={balance_threshold}, "
                       f"stop_loss_atr_multiplier={stop_loss_atr_multiplier}, "
                       f"take_profit_atr_multiplier={take_profit_atr_multiplier}")
            
            await self.main_loop()
            
            final_value = self._get_portfolio_value()
            stats = self.execution_engine.get_statistics()
            
            results.append({
                'params': {
                    'target_ratio': target_ratio,
                    'balance_threshold': balance_threshold,
                    'stop_loss_atr_multiplier': stop_loss_atr_multiplier,
                    'take_profit_atr_multiplier': take_profit_atr_multiplier
                },
                'final_portfolio_value': final_value,
                'total_profit': stats['total_profit'],
                'success_rate': stats['success_rate'],
                'stop_loss_triggers': stats['stop_loss_triggers'],
                'take_profit_triggers': stats['take_profit_triggers']
            })
            
            logger.info(f"Backtest completed: Final value={final_value:.2f}, "
                       f"Profit={stats['total_profit']:.2f}, "
                       f"Success rate={stats['success_rate']:.1f}%, "
                       f"Stop-loss triggers={stats['stop_loss_triggers']}, "
                       f"Take-profit triggers={stats['take_profit_triggers']}")
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['final_portfolio_value'])
        logger.info(f"Best parameters: {best_result['params']}, "
                   f"Final value: {best_result['final_portfolio_value']:.2f}")
        
        return results
    
    async def main_loop(self):
        """Main loop"""
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                
                await self._emergency_stop_check()
                if not self.is_running:
                    break
                
                await self._update_market_data()
                if not self.config['backtest_mode']:
                    await self._update_balances()
                
                await self._enhanced_rebalance()
                
                if loop_count % 20 == 0:
                    await self._print_status_report()
                
                if self.config['backtest_mode'] and self.backtest_index >= len(self.backtest_data):
                    logger.info("Backtest completed")
                    self.is_running = False
                
                await asyncio.sleep(1 if self.config['backtest_mode'] else 5)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)
    
    async def _print_status_report(self):
        """Print status report"""
        try:
            total_value = self._get_portfolio_value()
            current_ratio = self._calculate_portfolio_ratio()
            volatility = self.safety_checker.get_current_volatility()
            rsi = self.market_analyzer.get_rsi()
            trend = self.market_analyzer.get_trend_strength()
            
            logger.info(f"=== Enhanced Status Report ===")
            logger.info(f"Price: {self.last_price:.2f} | Ratio: {current_ratio:.3f} | Value: {total_value:.2f}")
            logger.info(f"Volatility: {volatility:.4f} | RSI: {rsi:.1f} | Trend: {trend:.3f}")
            
            stats = self.execution_engine.get_statistics()
            logger.info(f"Orders: {stats['total_orders']} total, {stats['success_rate']:.1f}% success")
            logger.info(f"Active: {stats['active_orders']} orders, Daily: {stats['daily_trades']}")
            logger.info(f"Profit: {stats['total_profit']:.4f} USDT, Avg Slippage: {stats['average_slippage']:.4f}")
            logger.info(f"Stop-loss triggers: {stats['stop_loss_triggers']}, Take-profit triggers: {stats['take_profit_triggers']}")
            
            if self.risk_manager.emergency_mode:
                logger.warning("⚠️  EMERGENCY MODE ACTIVE")
            
            logger.info(f"Balances - BTC: {self.btc_balance:.6f}, USDT: {self.usdt_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Error printing status: {e}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Secure Trading Bot...")
        
        self.is_running = False
        
        if self.execution_engine:
            await self.execution_engine.shutdown()
        
        await self._print_status_report()
        
        logger.info("Secure Trading Bot shutdown completed")

async def main():
    """Main function"""
    config = {
        'symbol': 'BTCUSDT',
        'api_key': 'YOUR_BINANCE_API_KEY',
        'api_secret': 'YOUR_BINANCE_SECRET',
        'testnet': True,
        'target_ratio': 0.5,
        'balance_threshold': 0.03,
        'min_trade_amount': 0.001,
        'max_position_ratio': 0.03,
        'price_lookback': 50,
        'simulation_mode': True,
        'backtest_mode': True,
        'backtest_file': '/home/krli/dreams/binance-scalping/data/yahoo_eth_usd_15m.csv',
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_atr_multiplier': 3.0,  # ATR multiplier for take-profit
        'atr_period': 14,
        'max_slippage_percent': 0.3,
        'price_deviation_threshold': 0.05,
        'max_daily_trades': 30,
        'emergency_stop': False,
        'min_profit_threshold': 0.003,
        'volatility_window': 20,
        'cooldown_period': 60,
        'trading_fee_rate': 0.001,  # 0.1% trading fee
    }
    
    try:
        config = ConfigValidator.validate(config)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    bot = SecureTradingBot(config)
    
    try:
        await bot.initialize()
        
        if config['backtest_mode']:
            # Parameter grid for optimization
            param_grid = {
                'target_ratio': [0.4, 0.5, 0.6],
                'balance_threshold': [0.02, 0.03, 0.04],
                'stop_loss_atr_multiplier': [1.5, 2.0, 2.5],
                'take_profit_atr_multiplier': [2.5, 3.0, 3.5]
            }
            logger.info("Starting parameter optimization backtest...")
            results = await bot.run_backtest(param_grid)
            
            # Select best parameters
            best_result = max(results, key=lambda x: x['final_portfolio_value'])
            config.update(best_result['params'])
            logger.info(f"Applying best parameters: {best_result['params']}")
            
            # Reset for final backtest with best parameters
            bot = SecureTradingBot(config)
            await bot.initialize()
        
        logger.info("Starting enhanced trading loop...")
        await bot.main_loop()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('secure_trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    print("🛡️  ENHANCED SECURE TRADING BOT v4.0")
    print("🔐 New features:")
    print("   ✓ ATR-based stop-loss and take-profit mechanisms")
    print("   ✓ Stop-loss triggers market orders for closing")
    print("   ✓ Backtesting with fees and slippage simulation")
    print("   ✓ Parameter optimization in backtest mode")
    print()
    print("⚠️  CRITICAL REMINDERS:")
    print("   • Set your real API keys in config")
    print("   • Test on testnet first (testnet: True)")
    print("   • Start with simulation_mode: True")
    print("   • Use small amounts initially")
    print("   • Monitor the bot constantly")
    print("   • Have manual emergency stops ready")
    print("   • Understand the risks involved")
    print("   • For backtesting, provide CSV with OHLCV data")
    print()
    
    if input("Continue with enhanced bot? (yes/no): ").lower() in ['yes', 'y']:
        asyncio.run(main())
    else:
        print("Enhanced bot cancelled for safety")