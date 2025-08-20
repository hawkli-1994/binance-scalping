import asyncio
import time
import uuid
import logging
import hmac
import hashlib
import aiohttp
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Union
from decimal import Decimal, ROUND_DOWN
import numpy as np
from collections import deque, defaultdict
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TradingError(Exception):
    """交易相关错误基类"""
    pass

class InsufficientBalanceError(TradingError):
    """余额不足错误"""
    pass

class InvalidPriceError(TradingError):
    """价格无效错误"""
    pass

class OrderRejectedError(TradingError):
    """订单被拒绝错误"""
    pass

class BinanceAPIError(TradingError):
    """Binance API错误"""
    pass

@dataclass
class ExchangeInfo:
    """交易所信息"""
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
    """Binance异步客户端"""
    
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
        """生成API签名"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _get_timestamp(self) -> int:
        """获取时间戳"""
        return int(time.time() * 1000)
    
    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        """发送API请求"""
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
        """获取交易所信息"""
        return await self._request('GET', '/api/v3/exchangeInfo')
    
    async def get_symbol_ticker(self, symbol: str):
        """获取交易对价格"""
        return await self._request('GET', '/api/v3/ticker/price', {'symbol': symbol})
    
    async def get_account(self):
        """获取账户信息"""
        return await self._request('GET', '/api/v3/account', signed=True)
    
    async def create_order(self, **params):
        """创建订单"""
        return await self._request('POST', '/api/v3/order', params, signed=True)
    
    async def get_order(self, symbol: str, orderId: str):
        """获取订单信息"""
        params = {'symbol': symbol, 'orderId': orderId}
        return await self._request('GET', '/api/v3/order', params, signed=True)
    
    async def cancel_order(self, symbol: str, orderId: str):
        """取消订单"""
        params = {'symbol': symbol, 'orderId': orderId}
        return await self._request('DELETE', '/api/v3/order', params, signed=True)
    
    async def get_open_orders(self, symbol: str = None):
        """获取当前挂单"""
        params = {'symbol': symbol} if symbol else {}
        return await self._request('GET', '/api/v3/openOrders', params, signed=True)
    
    async def get_klines(self, symbol: str, interval: str, limit: int = 500):
        """获取K线数据"""
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        return await self._request('GET', '/api/v3/klines', params)

class ConfigValidator:
    """配置验证器"""
    
    REQUIRED_KEYS = [
        'symbol', 'target_ratio', 'balance_threshold', 'burst_threshold',
        'min_trade_amount', 'max_position_ratio', 'price_lookback', 
        'api_key', 'api_secret'
    ]
    
    @classmethod
    def validate(cls, config: Dict) -> Dict:
        """验证配置"""
        # 检查必需键
        for key in cls.REQUIRED_KEYS:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # 验证API密钥
        if not config['api_key'] or not config['api_secret']:
            raise ValueError("API key and secret must not be empty")
        
        # 验证数值范围
        if not 0 < config['target_ratio'] < 1:
            raise ValueError("target_ratio must be between 0 and 1")
        
        if not 0 < config['balance_threshold'] < 0.5:
            raise ValueError("balance_threshold must be between 0 and 0.5")
        
        if config['burst_threshold'] <= 0:
            raise ValueError("burst_threshold must be positive")
        
        if config['min_trade_amount'] <= 0:
            raise ValueError("min_trade_amount must be positive")
        
        if not 0 < config['max_position_ratio'] <= 0.1:  # 限制为10%
            raise ValueError("max_position_ratio must be between 0 and 0.1")
        
        # 设置默认值
        defaults = {
            'testnet': True,  # 默认使用测试网
            'simulation_mode': True,
            'max_chase_attempts': 3,
            'chase_interval': 5.0,
            'max_order_age': 300.0,  # 5分钟
            'partial_fill_threshold': 0.1,
            'max_slippage_percent': 0.5,  # 0.5%
            'price_deviation_threshold': 0.05,  # 5%
            'max_daily_trades': 50,  # 降低日交易限制
            'emergency_stop': False,
            'max_concurrent_orders': 3,  # 降低并发订单数
            'order_cleanup_interval': 3600,  # 1小时
            'max_order_history': 1000,
            'min_profit_threshold': 0.002,  # 0.2%最小盈利阈值
            'volatility_window': 20,  # 波动率计算窗口
            'cooldown_period': 30,  # 交易冷却期(秒)
        }
        
        for key, value in defaults.items():
            config.setdefault(key, value)
        
        return config

class PrecisionManager:
    """精度管理器"""
    
    def __init__(self, exchange_info: ExchangeInfo):
        self.exchange_info = exchange_info
    
    def format_price(self, price: float) -> str:
        """格式化价格"""
        # 使用Decimal确保精度
        decimal_price = Decimal(str(price))
        tick_size = Decimal(str(self.exchange_info.tick_size))
        
        # 向下舍入到最近的tick_size
        rounded_price = (decimal_price // tick_size) * tick_size
        
        return format(rounded_price, f'.{self.exchange_info.price_precision}f')
    
    def format_quantity(self, quantity: float) -> str:
        """格式化数量"""
        decimal_qty = Decimal(str(quantity))
        step_size = Decimal(str(self.exchange_info.step_size))
        
        # 向下舍入到最近的step_size
        rounded_qty = (decimal_qty // step_size) * step_size
        
        return format(rounded_qty, f'.{self.exchange_info.qty_precision}f')
    
    def validate_order(self, side: str, quantity: float, price: float) -> bool:
        """验证订单参数"""
        # 检查数量范围
        if quantity < self.exchange_info.min_qty or quantity > self.exchange_info.max_qty:
            logger.warning(f"Quantity {quantity} out of range [{self.exchange_info.min_qty}, {self.exchange_info.max_qty}]")
            return False
        
        # 检查价格范围
        if price < self.exchange_info.min_price or price > self.exchange_info.max_price:
            logger.warning(f"Price {price} out of range [{self.exchange_info.min_price}, {self.exchange_info.max_price}]")
            return False
        
        # 检查最小名义价值
        notional = quantity * price
        if notional < self.exchange_info.min_notional:
            logger.warning(f"Notional {notional} below minimum {self.exchange_info.min_notional}")
            return False
        
        return True

class VolatilityCalculator:
    """波动率计算器"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
    
    def add_price(self, price: float):
        """添加价格数据"""
        if len(self.returns) > 0:
            last_price = self.returns[-1] if len(self.returns) > 0 else price
            return_val = (price - last_price) / last_price if last_price > 0 else 0
            self.returns.append(return_val)
        else:
            self.returns.append(0)
    
    def get_volatility(self) -> float:
        """获取波动率"""
        if len(self.returns) < 5:
            return 0.02  # 默认2%
        
        return np.std(list(self.returns)) * np.sqrt(len(self.returns))

class SafetyChecker:
    """安全检查器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.last_valid_price = None
        self.price_history = deque(maxlen=50)
        self.volatility_calc = VolatilityCalculator(config['volatility_window'])
        self.last_trade_time = 0
    
    def check_price_sanity(self, price: float) -> bool:
        """检查价格合理性"""
        if price <= 0:
            return False
        
        # 检查价格偏离
        if self.last_valid_price:
            deviation = abs(price - self.last_valid_price) / self.last_valid_price
            if deviation > self.config['price_deviation_threshold']:
                logger.warning(f"Price deviation too large: {deviation:.2%}")
                return False
        
        # 检查价格波动
        if len(self.price_history) >= 10:
            recent_avg = np.mean(list(self.price_history)[-10:])
            if abs(price - recent_avg) / recent_avg > 0.05:  # 5%偏离
                logger.warning(f"Price deviated from recent average: {price} vs {recent_avg}")
                return False
        
        self.price_history.append(price)
        self.volatility_calc.add_price(price)
        self.last_valid_price = price
        return True
    
    def check_balance_sufficiency(self, side: str, quantity: float, price: float, 
                                 btc_balance: float, usdt_balance: float) -> bool:
        """检查余额充足性"""
        if side == 'BUY':
            required_usdt = quantity * price * 1.002  # 加入手续费缓冲
            return usdt_balance >= required_usdt
        else:  # SELL
            return btc_balance >= quantity
    
    def check_cooldown(self) -> bool:
        """检查交易冷却期"""
        current_time = time.time()
        if current_time - self.last_trade_time < self.config['cooldown_period']:
            return False
        return True
    
    def update_trade_time(self):
        """更新交易时间"""
        self.last_trade_time = time.time()
    
    def get_current_volatility(self) -> float:
        """获取当前波动率"""
        return self.volatility_calc.get_volatility()

class TradingLock:
    """交易锁机制"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_strategies = set()
    
    @asynccontextmanager
    async def acquire(self, strategy_name: str):
        """获取交易锁"""
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
    """安全订单执行引擎"""
    
    def __init__(self, client: BinanceAsyncClient, config: Dict, exchange_info: ExchangeInfo):
        self.client = client
        self.config = ConfigValidator.validate(config)
        self.exchange_info = exchange_info
        self.precision_manager = PrecisionManager(exchange_info)
        self.safety_checker = SafetyChecker(self.config)
        self.trading_lock = TradingLock()
        
        # 订单管理
        self.active_orders: Dict[str, Dict] = {}
        self.completed_orders = deque(maxlen=self.config['max_order_history'])
        self.order_cleanup_task = None
        
        # 统计信息
        self.stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'rejected_orders': 0,
            'cancelled_orders': 0,
            'total_slippage': 0.0,
            'daily_trades': 0,
            'last_reset_date': time.time(),
            'total_profit': 0.0
        }
        
        # 启动清理任务
        if not self.config['simulation_mode']:
            self.order_cleanup_task = asyncio.create_task(self._cleanup_orders_periodically())
    
    async def _cleanup_orders_periodically(self):
        """定期清理订单"""
        while True:
            try:
                await asyncio.sleep(self.config['order_cleanup_interval'])
                await self._cleanup_old_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    async def _cleanup_old_orders(self):
        """清理旧订单"""
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
        """重置日统计"""
        current_date = time.time() // 86400  # 天数
        last_date = self.stats['last_reset_date'] // 86400
        
        if current_date != last_date:
            self.stats['daily_trades'] = 0
            self.stats['last_reset_date'] = time.time()
    
    def _can_trade(self) -> bool:
        """检查是否可以交易"""
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
        """安全下单"""
        if not self._can_trade():
            return None
        
        max_slippage_percent = max_slippage_percent or self.config['max_slippage_percent']
        
        try:
            # 验证价格
            if price and not self.safety_checker.check_price_sanity(price):
                raise InvalidPriceError(f"Price {price} failed sanity check")
            
            # 验证订单参数
            if not self.precision_manager.validate_order(side, quantity, price):
                raise OrderRejectedError("Order parameters validation failed")
            
            # 格式化精度
            formatted_qty = self.precision_manager.format_quantity(quantity)
            formatted_price = self.precision_manager.format_price(price) if price else None
            
            # 生成订单ID
            order_id = str(uuid.uuid4())
            client_order_id = f"bot_{int(time.time())}_{order_id[:8]}"
            
            # 模拟模式
            if self.config['simulation_mode']:
                return await self._simulate_order_secure(
                    order_id, symbol, side, float(formatted_qty), 
                    float(formatted_price) if formatted_price else None,
                    order_type, strategy_name
                )
            
            # 实际下单
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
            
            # 执行下单
            result = await self._safe_api_call(self.client.create_order, **order_params)
            
            if result:
                # 记录订单
                self.active_orders[order_id] = {
                    'exchange_order_id': result['orderId'],
                    'symbol': symbol,
                    'side': side,
                    'quantity': float(formatted_qty),
                    'price': float(formatted_price) if formatted_price else None,
                    'order_type': order_type,
                    'strategy': strategy_name,
                    'created_time': time.time(),
                    'status': 'NEW'
                }
                
                self.stats['total_orders'] += 1
                self.stats['daily_trades'] += 1
                self.safety_checker.update_trade_time()
                
                logger.info(f"Order placed: {order_id} - {side} {formatted_qty} @ {formatted_price}")
                
                # 启动监控
                asyncio.create_task(self._monitor_order_secure(order_id))
                
                return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.stats['rejected_orders'] += 1
            return None
    
    async def _simulate_order_secure(self, order_id: str, symbol: str, side: str, 
                                   quantity: float, price: float, order_type: str,
                                   strategy_name: str) -> str:
        """安全模拟下单"""
        # 模拟延迟
        await asyncio.sleep(np.random.uniform(0.1, 0.3))
        
        # 计算模拟滑点
        volatility = self.safety_checker.get_current_volatility()
        if order_type == 'MARKET':
            simulated_slippage = np.random.normal(0, volatility * 0.5)  # 基于波动率的滑点
            if abs(simulated_slippage) > self.config['max_slippage_percent'] / 100:
                logger.warning(f"Simulated slippage {simulated_slippage:.4f} exceeds limit")
                return None
            
            execution_price = price * (1 + simulated_slippage)
        else:
            execution_price = price
            
            # 限价单模拟成交概率（基于波动率）
            fill_probability = max(0.7, 0.95 - volatility * 10)  # 波动率越高，成交概率越低
            if np.random.random() > fill_probability:
                logger.info(f"Simulated order {order_id} not filled (volatility: {volatility:.4f})")
                return None
        
        # 记录模拟订单
        self.active_orders[order_id] = {
            'exchange_order_id': f"sim_{order_id}",
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'order_type': order_type,
            'strategy': strategy_name,
            'created_time': time.time(),
            'status': 'FILLED'
        }
        
        # 立即标记为完成
        await asyncio.sleep(0.1)
        self._complete_order(order_id, 'FILLED', execution_price)
        
        return order_id
    
    async def _monitor_order_secure(self, order_id: str):
        """安全订单监控"""
        monitor_start_time = time.time()
        max_monitor_time = self.config['max_order_age']
        
        while order_id in self.active_orders:
            try:
                # 检查监控超时
                if time.time() - monitor_start_time > max_monitor_time:
                    logger.warning(f"Order {order_id} monitoring timeout")
                    await self.cancel_order(order_id, reason="timeout")
                    break
                
                # 更新订单状态
                await self._update_order_status_secure(order_id)
                
                order_data = self.active_orders.get(order_id)
                if not order_data:
                    break
                
                # 检查是否完成
                if order_data['status'] in ['FILLED', 'CANCELED', 'REJECTED']:
                    self._complete_order(order_id, order_data['status'], 
                                       order_data.get('execution_price', order_data.get('price')))
                    break
                
                await asyncio.sleep(2)  # 2秒检查间隔
                
            except Exception as e:
                logger.error(f"Error monitoring order {order_id}: {e}")
                await asyncio.sleep(5)
    
    async def _update_order_status_secure(self, order_id: str):
        """安全更新订单状态"""
        if order_id not in self.active_orders:
            return
        
        order_data = self.active_orders[order_id]
        
        try:
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
                        # 使用成交均价
                        avg_price = float(result.get('cummulativeQuoteQty', 0)) / executed_qty
                        order_data['execution_price'] = avg_price
                    else:
                        order_data['execution_price'] = order_data['price']
                    
                    # 计算滑点
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
        """完成订单"""
        if order_id in self.active_orders:
            order_data = self.active_orders.pop(order_id)
            order_data['completed_time'] = time.time()
            order_data['final_status'] = status
            if execution_price:
                order_data['execution_price'] = execution_price
            
            self.completed_orders.append(order_data)
            
            # 更新统计
            if status == 'FILLED':
                self.stats['successful_orders'] += 1
                # 计算盈利（简化版）
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
        """取消订单"""
        if order_id not in self.active_orders:
            return False
        
        order_data = self.active_orders[order_id]
        
        try:
            if not self.config['simulation_mode']:
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
        """安全API调用"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return await func(**kwargs)
            except BinanceAPIError as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # 不应重试的错误
                if any(keyword in error_msg for keyword in [
                    'insufficient', 'invalid', 'unauthorized', 'forbidden',
                    'account has insufficient balance', 'filter failure'
                ]):
                    raise e
                
                # 可重试的错误
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.warning(f"API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Unexpected error after {max_retries} attempts: {e}")
        
        raise last_exception
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        success_rate = (self.stats['successful_orders'] / max(self.stats['total_orders'], 1)) * 100
        avg_slippage = self.stats['total_slippage'] / max(self.stats['successful_orders'], 1)
        
        return {
            'total_orders': self.stats['total_orders'],
            'successful_orders': self.stats['successful_orders'],
            'success_rate': success_rate,
            'average_slippage': avg_slippage,
            'daily_trades': self.stats['daily_trades'],
            'active_orders': len(self.active_orders),
            'total_profit': self.stats['total_profit']
        }
    
    async def shutdown(self):
        """关闭执行引擎"""
        # 取消清理任务
        if self.order_cleanup_task:
            self.order_cleanup_task.cancel()
            try:
                await self.order_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有活跃订单
        cancel_tasks = []
        for order_id in list(self.active_orders.keys()):
            cancel_tasks.append(self.cancel_order(order_id, reason="shutdown"))
        
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)
        
        logger.info("Order execution engine shutdown completed")


class RiskManager:
    """风险管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_drawdown = 0.05  # 5%最大回撤
        self.position_size_multiplier = 1.0
        self.emergency_mode = False
        
    def calculate_position_size(self, total_value: float, volatility: float, 
                              target_amount: float) -> float:
        """根据风险计算仓位大小"""
        # 基于波动率调整仓位
        volatility_adjustment = max(0.1, min(1.0, 1.0 - volatility * 10))
        
        # 基于回撤调整
        drawdown_adjustment = self.position_size_multiplier
        
        # 最终仓位
        adjusted_amount = target_amount * volatility_adjustment * drawdown_adjustment
        
        # 限制最大仓位
        max_position = total_value * self.config['max_position_ratio']
        
        return min(adjusted_amount, max_position)
    
    def update_drawdown(self, current_value: float, peak_value: float):
        """更新回撤信息"""
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > self.max_drawdown:
                self.position_size_multiplier *= 0.5  # 减半仓位
                logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}, reducing position size")
                
                if drawdown > self.max_drawdown * 2:
                    self.emergency_mode = True
                    logger.critical("Emergency mode activated due to excessive drawdown")
    
    def is_emergency_mode(self) -> bool:
        """检查是否为紧急模式"""
        return self.emergency_mode


class MarketAnalyzer:
    """市场分析器"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prices = deque(maxlen=window_size)
        self.volumes = deque(maxlen=window_size)
    
    def add_data(self, price: float, volume: float = 0):
        """添加市场数据"""
        self.prices.append(price)
        self.volumes.append(volume)
    
    def get_trend_strength(self) -> float:
        """获取趋势强度 (-1 到 1)"""
        if len(self.prices) < 20:
            return 0
        
        prices = list(self.prices)
        short_ma = np.mean(prices[-10:])
        long_ma = np.mean(prices[-20:])
        
        if long_ma == 0:
            return 0
        
        trend = (short_ma - long_ma) / long_ma
        return np.clip(trend * 100, -1, 1)  # 归一化到[-1, 1]
    
    def get_rsi(self, period: int = 14) -> float:
        """计算RSI指标"""
        if len(self.prices) < period + 1:
            return 50  # 中性值
        
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
        """检查是否超卖"""
        return self.get_rsi() < 30
    
    def is_overbought(self) -> bool:
        """检查是否超买"""
        return self.get_rsi() > 70


class SecureTradingBot:
    """安全交易机器人"""
    
    def __init__(self, config: Dict):
        self.config = ConfigValidator.validate(config)
        self.client = None
        self.execution_engine = None
        self.exchange_info = None
        
        # 状态变量
        self.is_running = False
        self.btc_balance = 0.0
        self.usdt_balance = 0.0
        self.last_price = 0.0
        self.price_history = deque(maxlen=self.config['price_lookback'])
        self.peak_portfolio_value = 0.0
        
        # 安全组件
        self.safety_checker = None
        self.trading_lock = TradingLock()
        self.risk_manager = RiskManager(self.config)
        self.market_analyzer = MarketAnalyzer()
        
    async def initialize(self):
        """初始化机器人"""
        logger.info("Initializing Secure Trading Bot...")
        
        # 创建客户端
        self.client = BinanceAsyncClient(
            self.config['api_key'],
            self.config['api_secret'],
            testnet=self.config['testnet']
        )
        
        # 获取交易所信息
        await self._load_exchange_info()
        
        # 初始化安全组件
        self.safety_checker = SafetyChecker(self.config)
        
        # 初始化执行引擎
        self.execution_engine = SecureOrderExecutionEngine(
            self.client, self.config, self.exchange_info
        )
        
        # 初始化数据
        await self._update_market_data()
        await self._update_balances()
        
        # 设置初始峰值
        self.peak_portfolio_value = self._get_portfolio_value()
        
        self.is_running = True
        logger.info("Secure Trading Bot initialized successfully")
    
    async def _load_exchange_info(self):
        """加载交易所信息"""
        try:
            async with self.client:
                exchange_info = await self.client.get_exchange_info()
                
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == self.config['symbol']:
                        filters = {f['filterType']: f for f in symbol_info['filters']}
                        
                        # 安全获取过滤器信息
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
    
    async def _update_market_data(self):
        """更新市场数据"""
        try:
            async with self.client:
                # 获取当前价格
                ticker = await self.client.get_symbol_ticker(symbol=self.config['symbol'])
                current_price = float(ticker['price'])
                
                # 安全检查
                if self.safety_checker.check_price_sanity(current_price):
                    self.last_price = current_price
                    self.price_history.append(current_price)
                    self.market_analyzer.add_data(current_price)
                else:
                    logger.warning(f"Price {current_price} failed safety check, using last valid price")
                    
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _update_balances(self):
        """更新账户余额"""
        try:
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
        """获取投资组合总价值"""
        return self.btc_balance * self.last_price + self.usdt_balance
    
    def _calculate_portfolio_ratio(self) -> float:
        """计算投资组合比例"""
        if self.last_price <= 0:
            return 0.5
        
        btc_value = self.btc_balance * self.last_price
        total_value = btc_value + self.usdt_balance
        
        return btc_value / total_value if total_value > 0 else 0.5
    
    def _calculate_safe_trade_amount(self, side: str, target_ratio: float = None) -> float:
        """计算安全交易数量"""
        total_value = self._get_portfolio_value()
        volatility = self.safety_checker.get_current_volatility()
        
        # 基础交易量
        base_amount = max(
            self.config['min_trade_amount'],
            total_value * 0.01  # 1%的基础仓位
        )
        
        # 风险调整
        risk_adjusted_amount = self.risk_manager.calculate_position_size(
            total_value, volatility, base_amount
        )
        
        if side == 'BUY':
            # 买入BTC
            available_usdt = self.usdt_balance * 0.995  # 留0.5%缓冲
            max_by_balance = available_usdt / self.last_price
            
            # 考虑最小名义价值
            min_by_notional = self.exchange_info.min_notional / self.last_price
            
            trade_amount = min(risk_adjusted_amount / self.last_price, max_by_balance)
            
            # 如果有目标比例，计算精确需求
            if target_ratio is not None:
                current_ratio = self._calculate_portfolio_ratio()
                if current_ratio < target_ratio:
                    needed_btc_value = total_value * (target_ratio - current_ratio)
                    needed_btc = needed_btc_value / self.last_price
                    trade_amount = min(trade_amount, needed_btc * 0.8)  # 80%执行，更保守
            
            return max(trade_amount, min_by_notional) if trade_amount >= self.exchange_info.min_qty else 0
        
        else:  # SELL
            # 卖出BTC
            available_btc = self.btc_balance * 0.995  # 留0.5%缓冲
            trade_amount = min(risk_adjusted_amount / self.last_price, available_btc)
            
            if target_ratio is not None:
                current_ratio = self._calculate_portfolio_ratio()
                if current_ratio > target_ratio:
                    excess_btc_value = total_value * (current_ratio - target_ratio)
                    excess_btc = excess_btc_value / self.last_price
                    trade_amount = min(trade_amount, excess_btc * 0.8)
            
            return trade_amount if trade_amount >= self.exchange_info.min_qty else 0
    
    async def _enhanced_rebalance(self):
        """增强的再平衡策略"""
        async with self.trading_lock.acquire("rebalance") as acquired:
            if not acquired:
                return
            
            current_ratio = self._calculate_portfolio_ratio()
            target_ratio = self.config['target_ratio']
            threshold = self.config['balance_threshold']
            
            # 市场分析
            trend_strength = self.market_analyzer.get_trend_strength()
            rsi = self.market_analyzer.get_rsi()
            
            # 根据市场情况调整目标比例
            adjusted_target = target_ratio
            if abs(trend_strength) > 0.3:  # 强趋势
                if trend_strength > 0 and not self.market_analyzer.is_overbought():
                    adjusted_target = min(target_ratio + 0.1, 0.7)  # 上涨趋势，增加BTC比例
                elif trend_strength < 0 and not self.market_analyzer.is_oversold():
                    adjusted_target = max(target_ratio - 0.1, 0.3)  # 下跌趋势，减少BTC比例
            
            deviation = abs(current_ratio - adjusted_target)
            
            # 只有偏离足够大时才交易
            if deviation <= threshold:
                return
            
            logger.info(f"Enhanced rebalancing: {current_ratio:.3f} -> {adjusted_target:.3f} (trend: {trend_strength:.3f}, RSI: {rsi:.1f})")
            
            if current_ratio < adjusted_target - threshold:
                # 需要买入BTC
                trade_amount = self._calculate_safe_trade_amount('BUY', adjusted_target)
                if trade_amount >= self.exchange_info.min_qty:
                    if self.safety_checker.check_balance_sufficiency(
                        'BUY', trade_amount, self.last_price, 
                        self.btc_balance, self.usdt_balance
                    ):
                        # 使用稍微激进的价格以提高成交率
                        buy_price = self.last_price * 1.001  # 高0.1%
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
                # 需要卖出BTC
                trade_amount = self._calculate_safe_trade_amount('SELL', adjusted_target)
                if trade_amount >= self.exchange_info.min_qty:
                    if self.safety_checker.check_balance_sufficiency(
                        'SELL', trade_amount, self.last_price,
                        self.btc_balance, self.usdt_balance
                    ):
                        # 使用稍微保守的价格
                        sell_price = self.last_price * 0.999  # 低0.1%
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
    
    async def _smart_burst_strategy(self):
        """智能突发策略"""
        async with self.trading_lock.acquire("burst") as acquired:
            if not acquired:
                return
            
            if len(self.price_history) < self.config['price_lookback']:
                return
            
            prices = list(self.price_history)
            volatility = self.safety_checker.get_current_volatility()
            
            # 动态调整突发阈值
            dynamic_threshold = self.config['burst_threshold'] * (1 + volatility * 2)
            
            sma = np.mean(prices)
            std = np.std(prices)
            
            if std <= 0:
                return
            
            upper_bound = sma + std * dynamic_threshold
            lower_bound = sma - std * dynamic_threshold
            
            # 确保有足够的价格偏离和盈利空间
            min_deviation = self.config['min_profit_threshold']
            
            # 检查RSI确认信号
            rsi = self.market_analyzer.get_rsi()
            
            if self.last_price > upper_bound and rsi > 70:  # 超买确认
                deviation = (self.last_price - upper_bound) / sma
                if deviation > min_deviation:
                    # 卖出信号
                    trade_amount = self._calculate_safe_trade_amount('SELL')
                    if trade_amount >= self.exchange_info.min_qty:
                        if self.safety_checker.check_balance_sufficiency(
                            'SELL', trade_amount, self.last_price,
                            self.btc_balance, self.usdt_balance
                        ):
                            logger.info(f"Smart burst sell: price {self.last_price:.2f} > upper {upper_bound:.2f}, RSI: {rsi:.1f}")
                            await self.execution_engine.place_order_secure(
                                symbol=self.config['symbol'],
                                side='SELL',
                                quantity=trade_amount,
                                price=self.last_price,
                                order_type='LIMIT',
                                strategy_name='smart_burst_sell'
                            )
            
            elif self.last_price < lower_bound and rsi < 30:  # 超卖确认
                deviation = (lower_bound - self.last_price) / sma
                if deviation > min_deviation:
                    # 买入信号
                    trade_amount = self._calculate_safe_trade_amount('BUY')
                    if trade_amount >= self.exchange_info.min_qty:
                        if self.safety_checker.check_balance_sufficiency(
                            'BUY', trade_amount, self.last_price,
                            self.btc_balance, self.usdt_balance
                        ):
                            logger.info(f"Smart burst buy: price {self.last_price:.2f} < lower {lower_bound:.2f}, RSI: {rsi:.1f}")
                            await self.execution_engine.place_order_secure(
                                symbol=self.config['symbol'],
                                side='BUY',
                                quantity=trade_amount,
                                price=self.last_price,
                                order_type='LIMIT',
                                strategy_name='smart_burst_buy'
                            )
    
    async def _emergency_stop_check(self):
        """紧急停止检查"""
        # 检查配置标志
        if self.config['emergency_stop']:
            logger.critical("Emergency stop activated by config")
            self.is_running = False
            return
        
        # 检查风险管理器
        if self.risk_manager.is_emergency_mode():
            logger.critical("Emergency mode activated by risk manager")
            self.is_running = False
            return
        
        # 检查余额异常
        total_value = self._get_portfolio_value()
        if total_value <= 0:
            logger.critical("Zero portfolio value detected - emergency stop")
            self.is_running = False
            return
        
        # 检查价格异常
        if self.last_price <= 0:
            logger.critical("Invalid price detected - emergency stop")
            self.is_running = False
            return
        
        # 更新风险管理
        if total_value > self.peak_portfolio_value:
            self.peak_portfolio_value = total_value
        else:
            self.risk_manager.update_drawdown(total_value, self.peak_portfolio_value)
    
    async def main_loop(self):
        """主循环"""
        loop_count = 0
        
        while self.is_running:
            try:
                loop_count += 1
                
                # 紧急停止检查
                await self._emergency_stop_check()
                if not self.is_running:
                    break
                
                # 更新数据
                await self._update_market_data()
                await self._update_balances()
                
                # 执行策略
                await self._enhanced_rebalance()
                await self._smart_burst_strategy()
                
                # 定期状态报告
                if loop_count % 20 == 0:
                    await self._print_status_report()
                
                # 休眠
                await asyncio.sleep(5)  # 增加到5秒间隔，减少API调用频率
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(30)  # 错误后等待更长时间
    
    async def _print_status_report(self):
        """打印状态报告"""
        try:
            total_value = self._get_portfolio_value()
            current_ratio = self._calculate_portfolio_ratio()
            volatility = self.safety_checker.get_current_volatility()
            rsi = self.market_analyzer.get_rsi()
            trend = self.market_analyzer.get_trend_strength()
            
            # 基础状态
            logger.info(f"=== Enhanced Status Report ===")
            logger.info(f"Price: {self.last_price:.2f} | Ratio: {current_ratio:.3f} | Value: {total_value:.2f}")
            logger.info(f"Volatility: {volatility:.4f} | RSI: {rsi:.1f} | Trend: {trend:.3f}")
            
            # 执行统计
            stats = self.execution_engine.get_statistics()
            logger.info(f"Orders: {stats['total_orders']} total, {stats['success_rate']:.1f}% success")
            logger.info(f"Active: {stats['active_orders']} orders, Daily: {stats['daily_trades']}")
            logger.info(f"Profit: {stats['total_profit']:.4f} USDT, Avg Slippage: {stats['average_slippage']:.4f}")
            
            # 风险状态
            if self.risk_manager.emergency_mode:
                logger.warning("⚠️  EMERGENCY MODE ACTIVE")
            
            # 余额详情
            logger.info(f"Balances - BTC: {self.btc_balance:.6f}, USDT: {self.usdt_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Error printing status: {e}")
    
    async def shutdown(self):
        """优雅关闭"""
        logger.info("Shutting down Secure Trading Bot...")
        
        self.is_running = False
        
        # 关闭执行引擎
        if self.execution_engine:
            await self.execution_engine.shutdown()
        
        # 最终状态报告
        await self._print_status_report()
        
        logger.info("Secure Trading Bot shutdown completed")


# 使用示例和测试
async def main():
    """主函数"""
    config = {
        'symbol': 'BTCUSDT',
        'api_key': 'YOUR_BINANCE_API_KEY',  # 替换为实际API密钥
        'api_secret': 'YOUR_BINANCE_SECRET',  # 替换为实际API密钥
        'testnet': True,  # 使用测试网，实盘交易时设为False
        'target_ratio': 0.5,
        'balance_threshold': 0.03,  # 稍微增加阈值，减少频繁交易
        'burst_threshold': 2.0,
        'min_trade_amount': 0.001,
        'max_position_ratio': 0.03,  # 降低为3%更安全
        'price_lookback': 50,
        'simulation_mode': True,  # 建议先模拟测试
        'max_slippage_percent': 0.3,
        'price_deviation_threshold': 0.05,
        'max_daily_trades': 30,  # 进一步减少
        'emergency_stop': False,
        'min_profit_threshold': 0.003,  # 0.3%最小盈利阈值
        'volatility_window': 20,
        'cooldown_period': 60,  # 增加到60秒冷却期
    }
    
    # 验证配置
    try:
        config = ConfigValidator.validate(config)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    bot = SecureTradingBot(config)
    
    try:
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
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('secure_trading_bot.log'),
            logging.StreamHandler()
        ]
    )
    
    # 安全提醒
    print("🛡️  ENHANCED SECURE TRADING BOT v2.0")
    print("🔐 New security features:")
    print("   ✓ Real Binance API integration")
    print("   ✓ Enhanced risk management")
    print("   ✓ Smart market analysis (RSI, trend)")
    print("   ✓ Dynamic position sizing")
    print("   ✓ Volatility-based adjustments")
    print("   ✓ Improved error handling")
    print("   ✓ Emergency stop mechanisms")
    print("   ✓ Profit threshold validation")
    print()
    print("⚠️  CRITICAL REMINDERS:")
    print("   • Set your real API keys in config")
    print("   • Test on testnet first (testnet: True)")
    print("   • Start with simulation_mode: True")
    print("   • Use small amounts initially")
    print("   • Monitor the bot constantly")
    print("   • Have manual emergency stops ready")
    print("   • Understand the risks involved")
    print()
    print("📊 Enhanced features:")
    print("   • Market trend analysis")
    print("   • RSI-based confirmations")
    print("   • Dynamic rebalancing")
    print("   • Volatility-adjusted position sizing")
    print("   • Improved profit tracking")
    print()
    
    if input("Continue with enhanced bot? (yes/no): ").lower() in ['yes', 'y']:
        asyncio.run(main())
    else:
        print("Enhanced bot cancelled for safety").error(f"API call failed after {max_retries} attempts: {e}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Unexpected error (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger