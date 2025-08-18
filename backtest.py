#!/usr/bin/env python3
"""
Backtesting module for the scalping strategy using historical data.
"""

import csv
import argparse
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional
import pytz
import copy


@dataclass
class Candle:
    """Represents a single candlestick data point."""
    timestamp: datetime
    price: float
    close: float
    high: float
    low: float
    open: float
    volume: float


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    order_type: str  # OPEN or CLOSE
    price: float
    quantity: float
    order_id: str
    status: str
    counter_order_id: Optional[str] = None


@dataclass
class BacktestConfig:
    """Configuration for backtesting.
    
    This class defines the configuration parameters used for backtesting
    the scalping strategy.
    
    Attributes:
        symbol: Trading pair symbol to backtest
        quantity: Order quantity for each trade
        take_profit: Take profit value in absolute price terms (USDC) for closing positions
        direction: Trading direction, either 'BUY' or 'SELL'
        max_orders: Maximum number of orders allowed in backtest
        wait_time: Time interval between checking market conditions
    """
    # 交易对符号，表示要交易的币种对，如BTCUSDC表示比特币兑USDC稳定币
    symbol: str = 'BTCUSDC'
    # 每笔交易的数量，表示每次下单的币种数量
    quantity: float = 0.01
    # 止盈点数，表示平仓获利的绝对价格值（以USDC为单位）表示的是单个币的价格变动值，而不是每个仓位的整体盈利值。
    take_profit: float = 1.0
    # 交易方向，表示开仓方向，可以是'BUY'（做多）或'SELL'（做空）
    direction: str = 'BUY'
    # 最大订单数，表示在回测中允许的最大订单数量
    max_orders: int = 75
    # 等待时间，表示检查市场条件的时间间隔（秒）
    wait_time: int = 30

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'BUY' if self.direction == "SELL" else 'SELL'


class BacktestEngine:
    """Backtesting engine for the scalping strategy."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.candles: List[Candle] = []
        self.trades: List[Trade] = []
        self.active_orders: Dict[str, float] = {}  # order_id -> price
        self.filled_orders: Dict[str, float] = {}  # order_id -> price
        self.cumulative_pnl = 0.0
        self.last_order_time = 0
        self.order_counter = 0
        
        # For performance metrics
        self.initial_capital = 0.0
        self.peak_capital = 0.0
        self.max_drawdown = 0.0
        self.capital_history = []  # List of (timestamp, capital)
        
    def load_data(self, csv_file_path: str):
        """Load candle data from CSV file."""
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            # Skip header rows
            next(reader)  # Price,Close,High,Low,Open,Volume
            next(reader)  # Ticker,BTC-USD,BTC-USD,BTC-USD,BTC-USD,BTC-USD
            next(reader)  # Datetime,,,,,
            
            for row in reader:
                if not row[0]:  # Skip empty rows
                    continue
                    
                timestamp = datetime.fromisoformat(row[0].replace('+00:00', ''))
                candle = Candle(
                    timestamp=timestamp,
                    price=float(row[1]),
                    close=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    open=float(row[4]),
                    volume=float(row[5]) if row[5] else 0.0
                )
                self.candles.append(candle)
        
        # Sort by timestamp
        self.candles.sort(key=lambda x: x.timestamp)
        print(f"Loaded {len(self.candles)} candles from {csv_file_path}")

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self.order_counter += 1
        return f"ORDER_{self.order_counter}"

    def _place_open_order(self, timestamp: datetime, price: float) -> str:
        """Place an open order at the specified price."""
        order_id = self._generate_order_id()
        
        # In real trading, we would place a LIMIT order at the best queue position
        # For backtesting, we'll assume the order gets filled at the next candle
        # if the price is favorable
        self.active_orders[order_id] = price
        self.trades.append(Trade(
            timestamp=timestamp,
            order_type="OPEN",
            price=price,
            quantity=self.config.quantity,
            order_id=order_id,
            status="PLACED"
        ))
        
        return order_id

    def _place_close_order(self, timestamp: datetime, open_price: float, open_order_id: str) -> str:
        """Place a close order."""
        close_price = open_price + (self.config.take_profit if self.config.direction == "BUY" 
                                   else -self.config.take_profit)
        order_id = self._generate_order_id()
        
        self.active_orders[order_id] = close_price
        self.trades.append(Trade(
            timestamp=timestamp,
            order_type="CLOSE",
            price=close_price,
            quantity=self.config.quantity,
            order_id=order_id,
            status="PLACED",
            counter_order_id=open_order_id
        ))
        
        return order_id

    def _process_candle(self, candle: Candle):
        """Process a single candle."""
        current_time = candle.timestamp.timestamp()
        
        # Check if any active orders would be filled based on this candle
        # Create a copy to avoid "dictionary changed size during iteration" error
        active_orders_copy = copy.deepcopy(self.active_orders)
        filled_orders = []
        
        for order_id, order_price in active_orders_copy.items():
            # Check if order would be filled
            if self._would_order_fill(order_price, candle):
                filled_orders.append(order_id)
                
                # Update trade status
                for trade in self.trades:
                    if trade.order_id == order_id:
                        trade.status = "FILLED"
                        trade.timestamp = candle.timestamp
                        break
                
                # Handle order fill
                if self._is_open_order(order_id):
                    # Open order filled, place close order
                    self.filled_orders[order_id] = order_price
                    close_order_id = self._place_close_order(candle.timestamp, order_price, order_id)
                    print(f"[{candle.timestamp}] Open order {order_id} filled at {order_price}, "
                          f"placed close order {close_order_id} at {order_price + self.config.take_profit}")
                else:
                    # Close order filled, realize profit
                    profit = self.config.take_profit * self.config.quantity
                    self.cumulative_pnl += profit
                    print(f"[{candle.timestamp}] Close order filled, profit: {profit}, "
                          f"cumulative PnL: {self.cumulative_pnl}")
                    
                    # Record capital for drawdown calculation
                    current_capital = self.initial_capital + self.cumulative_pnl
                    self.capital_history.append((candle.timestamp, current_capital))
                    
                    # Update peak capital and max drawdown
                    if current_capital > self.peak_capital:
                        self.peak_capital = current_capital
                    drawdown = (self.peak_capital - current_capital) / self.peak_capital if self.peak_capital > 0 else 0
                    if drawdown > self.max_drawdown:
                        self.max_drawdown = drawdown
                    
                    # Remove the corresponding open order from filled orders
                    open_order_id = None
                    for trade in self.trades:
                        if trade.order_id == order_id and trade.counter_order_id:
                            open_order_id = trade.counter_order_id
                            break
                    
                    if open_order_id and open_order_id in self.filled_orders:
                        del self.filled_orders[open_order_id]
        
        # Remove filled orders from active orders
        for order_id in filled_orders:
            if order_id in self.active_orders:
                del self.active_orders[order_id]
        
        # Place new orders if conditions are met
        if (len(self.active_orders) < self.config.max_orders and 
            (self.last_order_time == 0 or current_time - self.last_order_time >= self.config.wait_time)):
            
            open_order_id = self._place_open_order(candle.timestamp, candle.open)
            self.last_order_time = current_time
            print(f"[{candle.timestamp}] Placed open order {open_order_id} at {candle.open}")

    def _would_order_fill(self, order_price: float, candle: Candle) -> bool:
        """Determine if an order would fill based on candle data."""
        # Simplified fill logic - in real market, this would depend on order book
        # For BUY orders, fill if order_price >= low price
        # For SELL orders, fill if order_price <= high price
        if self.config.direction == "BUY":
            return order_price >= candle.low
        else:  # SELL
            return order_price <= candle.high

    def _is_open_order(self, order_id: str) -> bool:
        """Check if an order is an open order."""
        for trade in self.trades:
            if trade.order_id == order_id:
                return trade.order_type == "OPEN"
        return False

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.candles:
            return
            
        start_time = self.candles[0].timestamp
        end_time = self.candles[-1].timestamp
        time_period = (end_time - start_time).total_seconds()
        
        # Convert to days, months, years
        days = time_period / (24 * 3600)
        months = days / 30
        years = days / 365
        
        # Calculate annualized return
        if self.initial_capital > 0 and years > 0:
            # 修正年化收益率计算
            # cumulative_pnl是以USDC为单位的绝对利润
            # initial_capital是初始投入资本(价格*数量)
            # 所以total_return应该是绝对利润与初始资本的比率
            total_return = self.cumulative_pnl / self.initial_capital
            annualized_return = total_return / years
        else:
            total_return = float('inf')  # Cannot calculate without initial capital
            annualized_return = float('inf')
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "period_days": days,
            "period_months": months,
            "period_years": years,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": self.max_drawdown
        }

    def run_backtest(self):
        """Run the backtest."""
        if not self.candles:
            print("No candle data loaded. Please load data first.")
            return
            
        print(f"Starting backtest with {len(self.candles)} candles")
        print(f"Strategy: {self.config.direction} {self.config.quantity} @{self.config.symbol} "
              f"TP: {self.config.take_profit} Max Orders: {self.config.max_orders}")
        
        # Set initial capital based on first candle price and order quantity
        first_price = self.candles[0].price
        self.initial_capital = first_price * self.config.quantity
        self.peak_capital = self.initial_capital
        self.capital_history.append((self.candles[0].timestamp, self.initial_capital))
        
        for i, candle in enumerate(self.candles):
            if i % 100 == 0:
                print(f"Processing candle {i}/{len(self.candles)}")
            self._process_candle(candle)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        # Check if metrics calculation was successful
        if metrics is None:
            print("Error: Failed to calculate metrics")
            return
            
        print("Backtest completed")
        print(f"Total trades: {len(self.trades)}")
        print(f"Initial capital (approx): ${self.initial_capital:.2f} (based on {self.config.quantity} "
              f"contracts at price ${first_price:.2f})")
        print(f"Cumulative PnL: {self.cumulative_pnl}")
        print(f"Final capital: {self.initial_capital + self.cumulative_pnl:.2f}")
        print(f"Time period: {metrics['period_days']:.1f} days ({metrics['period_months']:.1f} months, "
              f"{metrics['period_years']:.2f} years)")
        print(f"Total return: {metrics['total_return']*100:.2f}%" if metrics['total_return'] != float('inf') 
              else "Total return: Cannot calculate (no initial capital)")
        print(f"Annualized return: {metrics['annualized_return']*100:.2f}%" if metrics['annualized_return'] != float('inf') 
              else "Annualized return: Cannot calculate (no initial capital)")
        print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Active orders remaining: {len(self.active_orders)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Backtest the scalping strategy')
    parser.add_argument('--data-file', type=str, required=True,
                        help='Path to the CSV data file')
    parser.add_argument('--symbol', type=str, default='BTCUSDC',
                        help='Trading pair symbol (default: BTCUSDC)')
    parser.add_argument('--quantity', type=float, default=0.01,
                        help='Order quantity (default: 0.01)')
    parser.add_argument('--take-profit', type=float, default=1.0,
                        help='Take profit in USDC (default: 1.0)')
    parser.add_argument('--direction', type=str, default='BUY',
                        help='Direction of the bot (default: BUY)')
    parser.add_argument('--max-orders', type=int, default=75,
                        help='Maximum number of active orders (default: 75)')
    parser.add_argument('--wait-time', type=int, default=30,
                        help='Wait time between orders in seconds (default: 30)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = BacktestConfig(
        symbol=args.symbol,
        quantity=args.quantity,
        take_profit=args.take_profit,
        direction=args.direction,
        max_orders=args.max_orders,
        wait_time=args.wait_time
    )
    
    # Create and run backtest
    engine = BacktestEngine(config)
    engine.load_data(args.data_file)
    engine.run_backtest()


if __name__ == "__main__":
    main()