"""
Core Trading Engine with CCXT Integration

This module provides a unified trading interface that supports both live trading
via CCXT and backtesting with CSV data.
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Trading mode enumeration"""
    LIVE = "live"
    BACKTEST = "backtest"
    SIMULATION = "simulation"

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "new"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class ExchangeInfo:
    """Exchange information dataclass"""
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
    maker_fee: float = 0.001
    taker_fee: float = 0.001

@dataclass
class Order:
    """Order dataclass"""
    id: str
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    average_price: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    exchange_order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fee: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Balance:
    """Balance dataclass"""
    asset: str
    free: float = 0.0
    used: float = 0.0
    total: float = 0.0

@dataclass
class Ticker:
    """Ticker dataclass"""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: float

@dataclass
class Candle:
    """Candle dataclass"""
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class TradingClient(ABC):
    """Abstract base class for trading clients"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exchange_info: Optional[ExchangeInfo] = None
        self.is_connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to exchange"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        pass
    
    @abstractmethod
    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange information for symbol"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """Place order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders"""
        pass
    
    @abstractmethod
    async def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Candle]:
        """Get candle data"""
        pass

class CCXTTradingClient(TradingClient):
    """CCXT-based trading client"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange = None
        self.exchange_id = config.get('exchange_id', 'binance')
        self.testnet = config.get('testnet', True)
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.sandbox_mode = config.get('sandbox_mode', True)
        
    async def connect(self) -> bool:
        """Connect to exchange using CCXT"""
        try:
            import ccxt.async_support as ccxt
            
            # Create exchange instance
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            
            # Configure testnet/sandbox
            if self.testnet or self.sandbox_mode:
                if hasattr(self.exchange, 'set_sandbox_mode'):
                    self.exchange.set_sandbox_mode(True)
                elif hasattr(self.exchange, 'testnet'):
                    self.exchange.testnet = True
            
            # Load markets
            await self.exchange.load_markets()
            self.is_connected = True
            logger.info(f"Connected to {self.exchange_id} exchange")
            return True
            
        except ImportError:
            logger.error("CCXT not installed. Please install with: pip install ccxt")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_id}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from exchange"""
        if self.exchange:
            await self.exchange.close()
            self.is_connected = False
            logger.info("Disconnected from exchange")
    
    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange information for symbol"""
        if not self.is_connected:
            await self.connect()
        
        try:
            market = self.exchange.market(symbol)
            
            # Get precision and limits
            price_precision = market.get('precision', {}).get('price', 8)
            qty_precision = market.get('precision', {}).get('amount', 8)
            
            # Get limits
            limits = market.get('limits', {})
            min_qty = limits.get('amount', {}).get('min', 0.001)
            max_qty = limits.get('amount', {}).get('max', 1000000)
            min_price = limits.get('price', {}).get('min', 0.01)
            max_price = limits.get('price', {}).get('max', 1000000)
            min_notional = limits.get('cost', {}).get('min', 10)
            
            # Get fees
            maker_fee = market.get('maker', 0.001)
            taker_fee = market.get('taker', 0.001)
            
            # Get tick size and step size
            tick_size = 10 ** (-price_precision)
            step_size = 10 ** (-qty_precision)
            
            self.exchange_info = ExchangeInfo(
                symbol=symbol,
                base_asset=market['base'],
                quote_asset=market['quote'],
                min_qty=min_qty,
                max_qty=max_qty,
                min_price=min_price,
                max_price=max_price,
                min_notional=min_notional,
                price_precision=price_precision,
                qty_precision=qty_precision,
                tick_size=tick_size,
                step_size=step_size,
                maker_fee=maker_fee,
                taker_fee=taker_fee
            )
            
            return self.exchange_info
            
        except Exception as e:
            logger.error(f"Failed to get exchange info for {symbol}: {e}")
            raise
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker"""
        if not self.is_connected:
            await self.connect()
        
        try:
            ticker = await self.exchange.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol,
                bid=ticker.get('bid', 0),
                ask=ticker.get('ask', 0),
                last=ticker.get('last', 0),
                volume=ticker.get('baseVolume', 0),
                timestamp=ticker.get('timestamp', time.time() * 1000) / 1000
            )
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances"""
        if not self.is_connected:
            await self.connect()
        
        try:
            balance = await self.exchange.fetch_balance()
            balances = {}
            
            for asset, data in balance.items():
                if isinstance(data, dict) and 'free' in data:
                    balances[asset] = Balance(
                        asset=asset,
                        free=float(data.get('free', 0)),
                        used=float(data.get('used', 0)),
                        total=float(data.get('total', 0))
                    )
            
            return balances
            
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            raise
    
    async def place_order(self, order: Order) -> Order:
        """Place order"""
        if not self.is_connected:
            await self.connect()
        
        try:
            # Prepare order parameters
            params = {
                'symbol': order.symbol,
                'type': order.type.value,
                'side': order.side.value,
                'amount': order.quantity,
                'clientOrderId': order.client_order_id
            }
            
            if order.type == OrderType.LIMIT and order.price:
                params['price'] = order.price
            
            # Place order
            result = await self.exchange.create_order(**params)
            
            # Update order with result
            order.exchange_order_id = str(result.get('id'))
            order.status = OrderStatus(result.get('status', 'new').lower())
            order.updated_at = time.time()
            
            logger.info(f"Order placed: {order.side.value} {order.quantity} {order.symbol} @ {order.price}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = time.time()
            return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if not self.is_connected:
            await self.connect()
        
        try:
            result = await self.exchange.cancel_order(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status"""
        if not self.is_connected:
            await self.connect()
        
        try:
            result = await self.exchange.fetch_order(order_id)
            
            order = Order(
                id=order_id,
                symbol=result.get('symbol'),
                side=OrderSide(result.get('side', '').lower()),
                type=OrderType(result.get('type', '').lower()),
                quantity=float(result.get('amount', 0)),
                price=float(result.get('price', 0)),
                status=OrderStatus(result.get('status', 'new').lower()),
                filled_quantity=float(result.get('filled', 0)),
                average_price=float(result.get('price', 0)),
                exchange_order_id=str(result.get('id')),
                fee=float(result.get('fee', {}).get('cost', 0))
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders"""
        if not self.is_connected:
            await self.connect()
        
        try:
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            results = await self.exchange.fetch_open_orders(params)
            orders = []
            
            for result in results:
                order = Order(
                    id=str(result.get('id')),
                    symbol=result.get('symbol'),
                    side=OrderSide(result.get('side', '').lower()),
                    type=OrderType(result.get('type', '').lower()),
                    quantity=float(result.get('amount', 0)),
                    price=float(result.get('price', 0)),
                    status=OrderStatus(result.get('status', 'new').lower()),
                    filled_quantity=float(result.get('filled', 0)),
                    average_price=float(result.get('price', 0)),
                    exchange_order_id=str(result.get('id')),
                    fee=float(result.get('fee', {}).get('cost', 0))
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []
    
    async def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Candle]:
        """Get candle data"""
        if not self.is_connected:
            await self.connect()
        
        try:
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            candles = []
            
            for data in ohlcv:
                candle = Candle(
                    symbol=symbol,
                    timestamp=data[0] / 1000,  # Convert from milliseconds
                    open=float(data[1]),
                    high=float(data[2]),
                    low=float(data[3]),
                    close=float(data[4]),
                    volume=float(data[5])
                )
                candles.append(candle)
            
            return candles
            
        except Exception as e:
            logger.error(f"Failed to get candles for {symbol}: {e}")
            return []

class BacktestClient(TradingClient):
    """Backtest client for CSV-based testing"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.data_file = config.get('data_file')
        self.data = None
        self.current_index = 0
        self.balances = {}
        self.orders = {}
        self.filled_orders = []
        self.current_time = None
        
    async def connect(self) -> bool:
        """Load backtest data"""
        try:
            if not self.data_file:
                raise ValueError("No data file provided for backtest")
            
            # Load CSV data
            self.data = pd.read_csv(self.data_file)
            
            # Validate required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in self.data.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            
            # Initialize balances
            initial_balance = self.config.get('initial_balance', 10000)
            self.balances = {
                'USDT': Balance('USDT', initial_balance, 0, initial_balance),
                self.config.get('base_asset', 'BTC'): Balance(self.config.get('base_asset', 'BTC'), 0, 0, 0)
            }
            
            self.is_connected = True
            logger.info(f"Loaded backtest data: {len(self.data)} rows")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load backtest data: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from backtest"""
        self.is_connected = False
        logger.info("Backtest disconnected")
    
    async def get_exchange_info(self, symbol: str) -> ExchangeInfo:
        """Get exchange information for backtest"""
        return ExchangeInfo(
            symbol=symbol,
            base_asset=symbol[:-4] if symbol.endswith('USDT') else 'BTC',
            quote_asset='USDT',
            min_qty=0.001,
            max_qty=1000000,
            min_price=0.01,
            max_price=1000000,
            min_notional=10,
            price_precision=2,
            qty_precision=6,
            tick_size=0.01,
            step_size=0.001,
            maker_fee=0.001,
            taker_fee=0.001
        )
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get current ticker from backtest data"""
        if not self.is_connected or self.data is None:
            await self.connect()
        
        if self.current_index >= len(self.data):
            raise ValueError("No more data available")
        
        row = self.data.iloc[self.current_index]
        self.current_time = row['timestamp']
        
        return Ticker(
            symbol=symbol,
            bid=row['close'] * 0.999,  # Simulate bid/ask spread
            ask=row['close'] * 1.001,
            last=row['close'],
            volume=row['volume'],
            timestamp=row['timestamp']
        )
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get current balances"""
        return self.balances
    
    async def place_order(self, order: Order) -> Order:
        """Place order in backtest"""
        if not self.is_connected or self.data is None:
            await self.connect()
        
        try:
            # Simulate order execution
            current_row = self.data.iloc[self.current_index]
            current_price = current_row['close']
            
            # Determine execution price
            if order.type == OrderType.MARKET:
                execution_price = current_price
            elif order.type == OrderType.LIMIT and order.price:
                execution_price = order.price
            else:
                execution_price = current_price
            
            # Check if order can be filled
            can_fill = False
            if order.side == OrderSide.BUY and execution_price <= current_price * 1.001:
                can_fill = True
            elif order.side == OrderSide.SELL and execution_price >= current_price * 0.999:
                can_fill = True
            
            if can_fill:
                # Calculate fee
                fee_rate = 0.001  # 0.1% fee
                fee = order.quantity * execution_price * fee_rate
                
                # Update balances
                if order.side == OrderSide.BUY:
                    cost = order.quantity * execution_price + fee
                    if self.balances['USDT'].free >= cost:
                        self.balances['USDT'].free -= cost
                        self.balances['USDT'].used += cost
                        base_asset = order.symbol[:-4] if order.symbol.endswith('USDT') else 'BTC'
                        if base_asset not in self.balances:
                            self.balances[base_asset] = Balance(base_asset, 0, 0, 0)
                        self.balances[base_asset].free += order.quantity
                        self.balances[base_asset].total += order.quantity
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = order.quantity
                        order.average_price = execution_price
                        order.fee = fee
                    else:
                        order.status = OrderStatus.REJECTED
                else:  # SELL
                    base_asset = order.symbol[:-4] if order.symbol.endswith('USDT') else 'BTC'
                    if base_asset in self.balances and self.balances[base_asset].free >= order.quantity:
                        self.balances[base_asset].free -= order.quantity
                        self.balances[base_asset].used += order.quantity
                        proceeds = order.quantity * execution_price - fee
                        self.balances['USDT'].free += proceeds
                        self.balances['USDT'].total += proceeds
                        order.status = OrderStatus.FILLED
                        order.filled_quantity = order.quantity
                        order.average_price = execution_price
                        order.fee = fee
                    else:
                        order.status = OrderStatus.REJECTED
            else:
                order.status = OrderStatus.NEW
            
            order.updated_at = time.time()
            self.orders[order.id] = order
            
            if order.status == OrderStatus.FILLED:
                self.filled_orders.append(order)
            
            logger.info(f"Backtest order: {order.side.value} {order.quantity} {order.symbol} @ {execution_price} - {order.status.value}")
            return order
            
        except Exception as e:
            logger.error(f"Failed to place backtest order: {e}")
            order.status = OrderStatus.REJECTED
            order.updated_at = time.time()
            return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in backtest"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.CANCELLED
            order.updated_at = time.time()
            logger.info(f"Backtest order cancelled: {order_id}")
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> Order:
        """Get order status in backtest"""
        return self.orders.get(order_id, Order(id=order_id, symbol='', side=OrderSide.BUY, type=OrderType.MARKET, quantity=0))
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders in backtest"""
        open_orders = []
        for order in self.orders.values():
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)
        return open_orders
    
    async def get_candles(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Candle]:
        """Get candle data from backtest"""
        if not self.is_connected or self.data is None:
            await self.connect()
        
        start_idx = max(0, self.current_index - limit)
        end_idx = min(len(self.data), self.current_index)
        
        candles = []
        for i in range(start_idx, end_idx):
            row = self.data.iloc[i]
            candle = Candle(
                symbol=symbol,
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            candles.append(candle)
        
        return candles
    
    def advance_time(self) -> bool:
        """Advance to next time step in backtest"""
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            return True
        return False
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        total_trades = len(self.filled_orders)
        winning_trades = 0
        total_profit = 0
        total_fees = 0
        
        for order in self.filled_orders:
            total_fees += order.fee
        
        # Calculate portfolio value over time
        portfolio_values = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            usdt_value = sum(balance.free + balance.used for balance in self.balances.values() if balance.asset == 'USDT')
            # Simplified - should calculate based on current price
            portfolio_values.append(usdt_value)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'total_profit': total_profit,
            'total_fees': total_fees,
            'portfolio_values': portfolio_values,
            'filled_orders': self.filled_orders
        }

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = TradingMode(config.get('mode', 'simulation'))
        self.client: Optional[TradingClient] = None
        self.is_running = False
        self.strategies = {}
        self.order_history = []
        
    async def initialize(self) -> bool:
        """Initialize trading engine"""
        try:
            # Create appropriate client based on mode
            if self.mode == TradingMode.LIVE:
                self.client = CCXTTradingClient(self.config)
            elif self.mode == TradingMode.BACKTEST:
                self.client = BacktestClient(self.config)
            else:  # SIMULATION
                self.client = CCXTTradingClient(self.config)
                self.client.config['sandbox_mode'] = True
            
            # Connect to exchange
            if not await self.client.connect():
                raise Exception("Failed to connect to exchange")
            
            # Get exchange info
            symbol = self.config.get('symbol', 'BTCUSDT')
            self.exchange_info = await self.client.get_exchange_info(symbol)
            
            logger.info(f"Trading engine initialized in {self.mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading engine: {e}")
            return False
    
    async def start(self) -> None:
        """Start trading engine"""
        if not self.client or not self.client.is_connected:
            await self.initialize()
        
        self.is_running = True
        logger.info("Trading engine started")
        
        # Start main trading loop
        await self._trading_loop()
    
    async def stop(self) -> None:
        """Stop trading engine"""
        self.is_running = False
        if self.client:
            await self.client.disconnect()
        logger.info("Trading engine stopped")
    
    async def _trading_loop(self) -> None:
        """Main trading loop"""
        while self.is_running:
            try:
                # Update market data
                await self._update_market_data()
                
                # Execute strategies
                await self._execute_strategies()
                
                # Manage orders
                await self._manage_orders()
                
                # Sleep based on mode
                if self.mode == TradingMode.BACKTEST:
                    if not self.client.advance_time():
                        logger.info("Backtest completed")
                        break
                    await asyncio.sleep(0.01)  # Minimal delay for backtest
                else:
                    await asyncio.sleep(1)  # 1 second for live trading
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)
    
    async def _update_market_data(self) -> None:
        """Update market data"""
        # This would be implemented to update market data
        pass
    
    async def _execute_strategies(self) -> None:
        """Execute all registered strategies"""
        for strategy_name, strategy in self.strategies.items():
            try:
                await strategy.execute()
            except Exception as e:
                logger.error(f"Error in strategy {strategy_name}: {e}")
    
    async def _manage_orders(self) -> None:
        """Manage open orders"""
        try:
            open_orders = await self.client.get_open_orders()
            
            for order in open_orders:
                # Check if order needs to be cancelled or modified
                # This would implement order management logic
                pass
                
        except Exception as e:
            logger.error(f"Error managing orders: {e}")
    
    def add_strategy(self, name: str, strategy) -> None:
        """Add a trading strategy"""
        self.strategies[name] = strategy
        logger.info(f"Strategy added: {name}")
    
    def remove_strategy(self, name: str) -> None:
        """Remove a trading strategy"""
        if name in self.strategies:
            del self.strategies[name]
            logger.info(f"Strategy removed: {name}")
    
    async def place_order(self, order: Order) -> Order:
        """Place order through trading engine"""
        try:
            result = await self.client.place_order(order)
            self.order_history.append(result)
            return result
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        try:
            balances = await self.client.get_balance()
            ticker = await self.client.get_ticker(self.config.get('symbol', 'BTCUSDT'))
            
            # Calculate portfolio value
            total_value = 0
            for balance in balances.values():
                if balance.asset == 'USDT':
                    total_value += balance.total
                elif balance.asset == 'BTC':
                    total_value += balance.total * ticker.last
            
            return {
                'balances': balances,
                'ticker': ticker,
                'total_value': total_value,
                'mode': self.mode.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    async def get_backtest_results(self) -> Dict[str, Any]:
        """Get backtest results"""
        if isinstance(self.client, BacktestClient):
            return self.client.get_backtest_results()
        return {}