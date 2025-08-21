"""
Trading Strategy System

This module implements a strategy pattern system for different trading approaches.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from collections import deque

from trading_engine import Order, OrderSide, OrderType, TradingEngine, Candle, Ticker

logger = logging.getLogger(__name__)

@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    symbol: str
    enabled: bool = True
    max_position_size: float = 0.1
    min_trade_amount: float = 0.001
    max_daily_trades: int = 50
    cooldown_period: int = 60
    risk_management: bool = True
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.03
    trailing_stop: bool = False
    trailing_stop_pct: float = 0.01

class TradingStrategy(ABC):
    """Abstract base class for trading strategies"""
    
    def __init__(self, config: StrategyConfig, trading_engine: TradingEngine):
        self.config = config
        self.trading_engine = trading_engine
        self.is_active = False
        self.orders = {}
        self.trade_history = []
        self.daily_trades = 0
        self.last_trade_time = 0
        self.last_execution_time = 0
        
        # Market data
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.candles = deque(maxlen=100)
        
        # Strategy state
        self.current_position = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize strategy"""
        pass
    
    @abstractmethod
    async def execute(self) -> None:
        """Execute strategy logic"""
        pass
    
    @abstractmethod
    def get_signal(self) -> str:
        """Get trading signal (buy, sell, hold)"""
        pass
    
    async def start(self) -> None:
        """Start strategy"""
        if await self.initialize():
            self.is_active = True
            logger.info(f"Strategy {self.config.name} started")
    
    async def stop(self) -> None:
        """Stop strategy"""
        self.is_active = False
        await self._cancel_all_orders()
        logger.info(f"Strategy {self.config.name} stopped")
    
    async def _cancel_all_orders(self) -> None:
        """Cancel all open orders"""
        try:
            open_orders = await self.trading_engine.client.get_open_orders(self.config.symbol)
            for order in open_orders:
                await self.trading_engine.client.cancel_order(order.id)
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
    
    async def _place_order(self, side: OrderSide, quantity: float, price: Optional[float] = None, 
                          order_type: OrderType = OrderType.LIMIT, 
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Optional[Order]:
        """Place order with risk management"""
        try:
            # Check cooldown
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_trade_time < self.config.cooldown_period:
                logger.debug(f"Strategy {self.config.name} in cooldown period")
                return None
            
            # Check daily trade limit
            if self.daily_trades >= self.config.max_daily_trades:
                logger.warning(f"Daily trade limit reached for {self.config.name}")
                return None
            
            # Create order
            order = Order(
                id=f"{self.config.name}_{int(current_time * 1000)}",
                symbol=self.config.symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            # Place order through trading engine
            result = await self.trading_engine.place_order(order)
            
            if result.status.value in ['filled', 'partially_filled']:
                self.last_trade_time = current_time
                self.daily_trades += 1
                self.trade_history.append(result)
                
                # Update position
                if side == OrderSide.BUY:
                    self.current_position += result.filled_quantity
                    if self.entry_price == 0:
                        self.entry_price = result.average_price
                else:
                    self.current_position -= result.filled_quantity
                    if self.current_position <= 0:
                        self.entry_price = 0
                
                logger.info(f"Order executed: {side.value} {quantity} {self.config.symbol} @ {price}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _update_market_data(self, ticker: Ticker, candles: List[Candle]) -> None:
        """Update market data"""
        self.price_history.append(ticker.last)
        self.volume_history.append(ticker.volume)
        
        for candle in candles:
            self.candles.append(candle)
        
        # Update highest/lowest prices for trailing stop
        if self.current_position > 0:
            self.highest_price = max(self.highest_price, ticker.last)
            self.lowest_price = min(self.lowest_price, ticker.last)
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if not self.config.risk_management or self.current_position <= 0:
            return False
        
        # Fixed stop loss
        if self.config.stop_loss_pct > 0:
            if self.entry_price > 0:
                stop_loss_price = self.entry_price * (1 - self.config.stop_loss_pct)
                if current_price <= stop_loss_price:
                    return True
        
        # Trailing stop
        if self.config.trailing_stop and self.config.trailing_stop_pct > 0:
            trailing_stop_price = self.highest_price * (1 - self.config.trailing_stop_pct)
            if current_price <= trailing_stop_price:
                return True
        
        return False
    
    def _check_take_profit(self, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if not self.config.risk_management or self.current_position <= 0:
            return False
        
        if self.config.take_profit_pct > 0 and self.entry_price > 0:
            take_profit_price = self.entry_price * (1 + self.config.take_profit_pct)
            if current_price >= take_profit_price:
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'name': self.config.name,
            'is_active': self.is_active,
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'daily_trades': self.daily_trades,
            'total_trades': len(self.trade_history),
            'last_trade_time': self.last_trade_time,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price
        }

class MovingAverageStrategy(TradingStrategy):
    """Moving average crossover strategy"""
    
    def __init__(self, config: StrategyConfig, trading_engine: TradingEngine, 
                 fast_period: int = 5, slow_period: int = 15):
        super().__init__(config, trading_engine)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.last_signal = 'hold'
    
    async def initialize(self) -> bool:
        """Initialize moving average strategy"""
        try:
            # Get initial candle data
            candles = await self.trading_engine.client.get_candles(
                self.config.symbol, '1m', self.slow_period + 10
            )
            
            if len(candles) < self.slow_period:
                logger.warning(f"Insufficient data for {self.config.name}")
                return False
            
            # Calculate initial MAs
            self._update_indicators(candles)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.config.name}: {e}")
            return False
    
    async def execute(self) -> None:
        """Execute moving average strategy"""
        if not self.is_active:
            return
        
        try:
            # Get current market data
            ticker = await self.trading_engine.client.get_ticker(self.config.symbol)
            candles = await self.trading_engine.client.get_candles(self.config.symbol, '1m', self.slow_period)
            
            # Update market data and indicators
            self._update_market_data(ticker, candles)
            self._update_indicators(candles)
            
            # Check risk management
            if self._check_stop_loss(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            if self._check_take_profit(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            # Get trading signal
            signal = self.get_signal()
            
            # Execute signal
            if signal == 'buy' and self.last_signal != 'buy':
                quantity = await self._calculate_position_size(ticker.last)
                if quantity > 0:
                    await self._place_order(OrderSide.BUY, quantity, ticker.last * 1.001)
            
            elif signal == 'sell' and self.last_signal != 'sell':
                if self.current_position > 0:
                    await self._place_order(OrderSide.SELL, self.current_position, ticker.last * 0.999)
            
            self.last_signal = signal
            
        except Exception as e:
            logger.error(f"Error executing {self.config.name}: {e}")
    
    def _update_indicators(self, candles: List[Candle]) -> None:
        """Update moving average indicators"""
        if len(candles) < self.slow_period:
            return
        
        prices = [candle.close for candle in candles]
        
        # Calculate fast MA
        if len(prices) >= self.fast_period:
            self.fast_ma = np.mean(prices[-self.fast_period:])
        
        # Calculate slow MA
        if len(prices) >= self.slow_period:
            self.slow_ma = np.mean(prices[-self.slow_period:])
    
    def get_signal(self) -> str:
        """Get moving average crossover signal"""
        if self.fast_ma == 0 or self.slow_ma == 0:
            return 'hold'
        
        if self.fast_ma > self.slow_ma:
            return 'buy'
        elif self.fast_ma < self.slow_ma:
            return 'sell'
        else:
            return 'hold'
    
    async def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size"""
        try:
            account_info = await self.trading_engine.get_account_info()
            total_value = account_info.get('total_value', 0)
            
            # Calculate position size based on config
            max_position_value = total_value * self.config.max_position_size
            quantity = max_position_value / current_price
            
            # Apply minimum trade amount
            return max(quantity, self.config.min_trade_amount)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

class EnhancedMovingAverageStrategy(TradingStrategy):
    """Enhanced moving average crossover strategy with volume and RSI filters"""
    
    def __init__(self, config: StrategyConfig, trading_engine: TradingEngine, 
                 fast_period: int = 5, slow_period: int = 15, rsi_period: int = 14, 
                 volume_period: int = 20):
        super().__init__(config, trading_engine)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.volume_period = volume_period
        self.fast_ma = 0.0
        self.slow_ma = 0.0
        self.rsi = 0.0
        self.volume_ma = 0.0
        self.last_signal = 'hold'
    
    async def initialize(self) -> bool:
        """Initialize enhanced moving average strategy"""
        try:
            # Get initial candle data
            candles = await self.trading_engine.client.get_candles(
                self.config.symbol, '1m', max(self.slow_period, self.rsi_period, self.volume_period) + 10
            )
            
            if len(candles) < max(self.slow_period, self.rsi_period, self.volume_period):
                logger.warning(f"Insufficient data for {self.config.name}")
                return False
            
            # Calculate initial indicators
            self._update_indicators(candles)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.config.name}: {e}")
            return False
    
    async def execute(self) -> None:
        """Execute enhanced moving average strategy"""
        if not self.is_active:
            return
        
        try:
            # Get current market data
            ticker = await self.trading_engine.client.get_ticker(self.config.symbol)
            candles = await self.trading_engine.client.get_candles(
                self.config.symbol, '1m', max(self.slow_period, self.rsi_period, self.volume_period)
            )
            
            # Update market data and indicators
            self._update_market_data(ticker, candles)
            self._update_indicators(candles)
            
            # Check risk management
            if self._check_stop_loss(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            if self._check_take_profit(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            # Get enhanced trading signal
            signal = self.get_signal()
            
            # Execute signal with enhanced conditions
            if signal == 'buy' and self.last_signal != 'buy':
                if self._confirm_buy_signal(ticker):
                    quantity = await self._calculate_position_size(ticker.last)
                    if quantity > 0:
                        await self._place_order(OrderSide.BUY, quantity, ticker.last * 1.001)
            
            elif signal == 'sell' and self.last_signal != 'sell':
                if self._confirm_sell_signal(ticker):
                    if self.current_position > 0:
                        await self._place_order(OrderSide.SELL, self.current_position, ticker.last * 0.999)
            
            self.last_signal = signal
            
        except Exception as e:
            logger.error(f"Error executing {self.config.name}: {e}")
    
    def _update_indicators(self, candles: List[Candle]) -> None:
        """Update all indicators"""
        if len(candles) < max(self.slow_period, self.rsi_period, self.volume_period):
            return
        
        prices = [candle.close for candle in candles]
        volumes = [candle.volume for candle in candles]
        
        # Calculate moving averages
        if len(prices) >= self.fast_period:
            self.fast_ma = np.mean(prices[-self.fast_period:])
        
        if len(prices) >= self.slow_period:
            self.slow_ma = np.mean(prices[-self.slow_period:])
        
        # Calculate RSI
        if len(prices) >= self.rsi_period:
            self.rsi = self._calculate_rsi(prices)
        
        # Calculate volume MA
        if len(volumes) >= self.volume_period:
            self.volume_ma = np.mean(volumes[-self.volume_period:])
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI indicator"""
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate average gains and losses
        avg_gains = np.mean(gains[-self.rsi_period:])
        avg_losses = np.mean(losses[-self.rsi_period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_signal(self) -> str:
        """Get enhanced moving average crossover signal"""
        if self.fast_ma == 0 or self.slow_ma == 0:
            return 'hold'
        
        if self.fast_ma > self.slow_ma:
            return 'buy'
        elif self.fast_ma < self.slow_ma:
            return 'sell'
        else:
            return 'hold'
    
    def _confirm_buy_signal(self, ticker: Ticker) -> bool:
        """Confirm buy signal with additional filters"""
        # Volume confirmation
        if self.volume_ma > 0 and ticker.volume < self.volume_ma * 1.2:
            return False
        
        # RSI confirmation (not overbought)
        if self.rsi > 70:
            return False
        
        # RSI confirmation (some momentum)
        if self.rsi < 30:
            return False
        
        return True
    
    def _confirm_sell_signal(self, ticker: Ticker) -> bool:
        """Confirm sell signal with additional filters"""
        # Volume confirmation
        if self.volume_ma > 0 and ticker.volume < self.volume_ma * 1.2:
            return False
        
        # RSI confirmation (overbought)
        if self.rsi < 30:
            return False
        
        return True
    
    async def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size"""
        try:
            account_info = await self.trading_engine.get_account_info()
            total_value = account_info.get('total_value', 0)
            
            # Calculate position size based on config
            max_position_value = total_value * self.config.max_position_size
            quantity = max_position_value / current_price
            
            # Apply minimum trade amount
            return max(quantity, self.config.min_trade_amount)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion strategy"""
    
    def __init__(self, config: StrategyConfig, trading_engine: TradingEngine, 
                 period: int = 10, std_dev_threshold: float = 1.5):
        super().__init__(config, trading_engine)
        self.period = period
        self.std_dev_threshold = std_dev_threshold
        self.mean_price = 0.0
        self.std_dev = 0.0
        self.upper_band = 0.0
        self.lower_band = 0.0
    
    async def initialize(self) -> bool:
        """Initialize mean reversion strategy"""
        try:
            candles = await self.trading_engine.client.get_candles(
                self.config.symbol, '1m', self.period + 10
            )
            
            if len(candles) < self.period:
                logger.warning(f"Insufficient data for {self.config.name}")
                return False
            
            self._update_indicators(candles)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.config.name}: {e}")
            return False
    
    async def execute(self) -> None:
        """Execute mean reversion strategy"""
        if not self.is_active:
            return
        
        try:
            ticker = await self.trading_engine.client.get_ticker(self.config.symbol)
            candles = await self.trading_engine.client.get_candles(self.config.symbol, '1m', self.period)
            
            self._update_market_data(ticker, candles)
            self._update_indicators(candles)
            
            # Check risk management
            if self._check_stop_loss(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            if self._check_take_profit(ticker.last):
                await self._place_order(OrderSide.SELL, self.current_position, None, OrderType.MARKET)
                return
            
            # Get signal
            signal = self.get_signal()
            
            # Execute signal
            if signal == 'buy' and self.current_position <= 0:
                quantity = await self._calculate_position_size(ticker.last)
                if quantity > 0:
                    await self._place_order(OrderSide.BUY, quantity, ticker.last * 0.999)
            
            elif signal == 'sell' and self.current_position > 0:
                await self._place_order(OrderSide.SELL, self.current_position, ticker.last * 1.001)
            
        except Exception as e:
            logger.error(f"Error executing {self.config.name}: {e}")
    
    def _update_indicators(self, candles: List[Candle]) -> None:
        """Update Bollinger bands"""
        if len(candles) < self.period:
            return
        
        prices = [candle.close for candle in candles]
        self.mean_price = np.mean(prices)
        self.std_dev = np.std(prices)
        self.upper_band = self.mean_price + (self.std_dev * self.std_dev_threshold)
        self.lower_band = self.mean_price - (self.std_dev * self.std_dev_threshold)
    
    def get_signal(self) -> str:
        """Get mean reversion signal"""
        if self.mean_price == 0:
            return 'hold'
        
        current_price = self.price_history[-1] if self.price_history else 0
        
        if current_price <= self.lower_band:
            return 'buy'
        elif current_price >= self.upper_band:
            return 'sell'
        else:
            return 'hold'
    
    async def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size with volatility adjustment"""
        try:
            account_info = await self.trading_engine.get_account_info()
            total_value = account_info.get('total_value', 0)
            
            # Adjust position size based on volatility
            volatility_factor = 1.0 / (1.0 + self.std_dev)
            adjusted_max_size = self.config.max_position_size * volatility_factor
            
            max_position_value = total_value * adjusted_max_size
            quantity = max_position_value / current_price
            
            return max(quantity, self.config.min_trade_amount)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

class GridTradingStrategy(TradingStrategy):
    """Grid trading strategy"""
    
    def __init__(self, config: StrategyConfig, trading_engine: TradingEngine, 
                 grid_spacing: float = 0.005, grid_levels: int = 10):
        super().__init__(config, trading_engine)
        self.grid_spacing = grid_spacing
        self.grid_levels = grid_levels
        self.grid_orders = {}
        self.base_price = 0.0
        self.grid_center = 0.0
    
    async def initialize(self) -> bool:
        """Initialize grid trading strategy"""
        try:
            ticker = await self.trading_engine.client.get_ticker(self.config.symbol)
            self.base_price = ticker.last
            self.grid_center = self.base_price
            
            # Create initial grid
            await self._create_grid()
            return True
            
        except Exception as e:
            logger.error(f"Error initializing {self.config.name}: {e}")
            return False
    
    async def execute(self) -> None:
        """Execute grid trading strategy"""
        if not self.is_active:
            return
        
        try:
            ticker = await self.trading_engine.client.get_ticker(self.config.symbol)
            
            # Update grid if price moves significantly
            if abs(ticker.last - self.grid_center) > self.grid_spacing * self.grid_levels / 2:
                self.grid_center = ticker.last
                await self._rebalance_grid()
            
            # Check if any grid orders are triggered
            await self._check_grid_orders(ticker.last)
            
        except Exception as e:
            logger.error(f"Error executing {self.config.name}: {e}")
    
    async def _create_grid(self) -> None:
        """Create grid orders"""
        try:
            # Cancel existing grid orders
            await self._cancel_grid_orders()
            
            # Create buy orders below grid center
            for i in range(1, self.grid_levels + 1):
                price = self.grid_center * (1 - self.grid_spacing * i)
                quantity = await self._calculate_grid_quantity()
                
                if quantity > 0:
                    order = await self._place_order(OrderSide.BUY, quantity, price)
                    if order:
                        self.grid_orders[order.id] = {'price': price, 'side': 'buy', 'level': i}
            
            # Create sell orders above grid center
            for i in range(1, self.grid_levels + 1):
                price = self.grid_center * (1 + self.grid_spacing * i)
                quantity = await self._calculate_grid_quantity()
                
                if quantity > 0 and self.current_position >= quantity:
                    order = await self._place_order(OrderSide.SELL, quantity, price)
                    if order:
                        self.grid_orders[order.id] = {'price': price, 'side': 'sell', 'level': i}
            
        except Exception as e:
            logger.error(f"Error creating grid: {e}")
    
    async def _cancel_grid_orders(self) -> None:
        """Cancel all grid orders"""
        for order_id in list(self.grid_orders.keys()):
            try:
                await self.trading_engine.client.cancel_order(order_id)
                del self.grid_orders[order_id]
            except Exception as e:
                logger.error(f"Error cancelling grid order {order_id}: {e}")
    
    async def _check_grid_orders(self, current_price: float) -> None:
        """Check if grid orders are triggered"""
        try:
            open_orders = await self.trading_engine.client.get_open_orders(self.config.symbol)
            
            for order in open_orders:
                if order.id in self.grid_orders:
                    grid_info = self.grid_orders[order.id]
                    
                    # Check if order should be filled
                    if (grid_info['side'] == 'buy' and current_price <= grid_info['price']) or \
                       (grid_info['side'] == 'sell' and current_price >= grid_info['price']):
                        
                        # Order triggered, create opposite order
                        await self._create_opposite_order(grid_info)
                        
                        # Remove from grid orders
                        del self.grid_orders[order.id]
                        
        except Exception as e:
            logger.error(f"Error checking grid orders: {e}")
    
    async def _create_opposite_order(self, filled_grid_info: Dict[str, Any]) -> None:
        """Create opposite order when grid order is filled"""
        try:
            if filled_grid_info['side'] == 'buy':
                # Create sell order at higher price
                new_price = filled_grid_info['price'] * (1 + self.grid_spacing * 2)
                quantity = await self._calculate_grid_quantity()
                
                if quantity > 0:
                    order = await self._place_order(OrderSide.SELL, quantity, new_price)
                    if order:
                        self.grid_orders[order.id] = {
                            'price': new_price, 
                            'side': 'sell', 
                            'level': filled_grid_info['level'] + 2
                        }
            else:
                # Create buy order at lower price
                new_price = filled_grid_info['price'] * (1 - self.grid_spacing * 2)
                quantity = await self._calculate_grid_quantity()
                
                if quantity > 0:
                    order = await self._place_order(OrderSide.BUY, quantity, new_price)
                    if order:
                        self.grid_orders[order.id] = {
                            'price': new_price, 
                            'side': 'buy', 
                            'level': filled_grid_info['level'] - 2
                        }
                        
        except Exception as e:
            logger.error(f"Error creating opposite order: {e}")
    
    async def _rebalance_grid(self) -> None:
        """Rebalance grid around new center"""
        await self._cancel_grid_orders()
        await self._create_grid()
    
    async def _calculate_grid_quantity(self) -> float:
        """Calculate grid order quantity"""
        try:
            account_info = await self.trading_engine.get_account_info()
            total_value = account_info.get('total_value', 0)
            
            # Allocate equal amount to each grid level
            value_per_level = total_value * self.config.max_position_size / self.grid_levels
            quantity = value_per_level / self.grid_center
            
            return max(quantity, self.config.min_trade_amount)
            
        except Exception as e:
            logger.error(f"Error calculating grid quantity: {e}")
            return 0
    
    def get_signal(self) -> str:
        """Grid trading doesn't use traditional signals"""
        return 'hold'

class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self, trading_engine: TradingEngine):
        self.trading_engine = trading_engine
        self.strategies = {}
        self.active_strategies = set()
    
    def add_strategy(self, strategy: TradingStrategy) -> None:
        """Add a strategy"""
        self.strategies[strategy.config.name] = strategy
        logger.info(f"Strategy added: {strategy.config.name}")
    
    def remove_strategy(self, name: str) -> None:
        """Remove a strategy"""
        if name in self.strategies:
            strategy = self.strategies[name]
            if strategy.is_active:
                asyncio.create_task(strategy.stop())
            del self.strategies[name]
            self.active_strategies.discard(name)
            logger.info(f"Strategy removed: {name}")
    
    async def start_strategy(self, name: str) -> bool:
        """Start a strategy"""
        if name in self.strategies:
            strategy = self.strategies[name]
            if not strategy.is_active:
                await strategy.start()
                self.active_strategies.add(name)
                return True
        return False
    
    async def stop_strategy(self, name: str) -> bool:
        """Stop a strategy"""
        if name in self.strategies:
            strategy = self.strategies[name]
            if strategy.is_active:
                await strategy.stop()
                self.active_strategies.discard(name)
                return True
        return False
    
    async def start_all_strategies(self) -> None:
        """Start all enabled strategies"""
        for name, strategy in self.strategies.items():
            if strategy.config.enabled:
                await self.start_strategy(name)
    
    async def stop_all_strategies(self) -> None:
        """Stop all strategies"""
        for name in list(self.active_strategies):
            await self.stop_strategy(name)
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics for all strategies"""
        stats = {}
        for name, strategy in self.strategies.items():
            stats[name] = strategy.get_statistics()
        return stats
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies"""
        return list(self.active_strategies)