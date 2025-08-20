
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
import pandas as pd

@dataclass
class Candle:
    """Represents a single candlestick data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
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
    """Configuration for backtesting."""
    symbol: str = 'BTCUSDC'
    quantity: float = 0.01
    take_profit: float = 1.0
    direction: str = 'BUY'
    max_orders: int = 75
    wait_time: int = 30  # In seconds, adjusted for candle timeframe
    fee_rate: float = 0 # 0.02 / 100  # Binance taker fee

    @property
    def close_order_side(self) -> str:
        """Get the close order side based on bot direction."""
        return 'BUY' if self.direction == "SELL" else 'SELL'

    def validate(self):
        """Validate configuration parameters."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.take_profit <= 0:
            raise ValueError("Take profit must be positive")
        if self.max_orders <= 0:
            raise ValueError("Max orders must be positive")
        if self.direction not in ["BUY", "SELL"]:
            raise ValueError("Direction must be 'BUY' or 'SELL'")

class BacktestEngine:
    """Backtesting engine for the scalping strategy."""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.config.validate()  # Validate config
        self.candles: List[Candle] = []
        self.trades: List[Trade] = []
        self.active_orders: Dict[str, float] = {}  # order_id -> price
        self.filled_orders: Dict[str, float] = {}  # order_id -> price
        self.cumulative_pnl = 0.0
        self.last_order_time = 0
        self.order_counter = 0
        self.initial_capital = 0.0
        self.peak_capital = 0.0
        self.max_drawdown = 0.0
        self.capital_history = []  # List of (timestamp, capital)

    def load_data(self, csv_file_path: str):
        """Load candle data from CSV file."""
        try:
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
                        open=float(row[4]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[1]),
                        volume=float(row[5]) if row[5] else 0.0
                    )
                    self.candles.append(candle)
            
            # Sort by timestamp and validate
            self.candles.sort(key=lambda x: x.timestamp)
            if not self.candles:
                raise ValueError("No valid candle data loaded")
            print(f"Loaded {len(self.candles)} candles from {csv_file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def _generate_order_id(self) -> str:
        """Generate a unique order ID."""
        self.order_counter += 1
        return f"ORDER_{self.order_counter}"

    def _place_open_order(self, timestamp: datetime, price: float) -> str:
        """Place an open order at the specified price."""
        order_id = self._generate_order_id()
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

    def _would_order_fill(self, order_id: str, order_price: float, candle: Candle) -> bool:
        """Determine if an order would fill based on candle data."""
        if self._is_open_order(order_id):
            if self.config.direction == "BUY":
                return candle.low <= order_price  # Market falls to or below buy price
            else:  # SELL
                return candle.high >= order_price  # Market rises to or above sell price
        else:  # Close order
            if self.config.close_order_side == "BUY":
                return candle.low <= order_price  # Market falls to close buy price
            else:  # SELL
                return candle.high >= order_price  # Market rises to close sell price

    def _is_open_order(self, order_id: str) -> bool:
        """Check if an order is an open order."""
        for trade in self.trades:
            if trade.order_id == order_id:
                return trade.order_type == "OPEN"
        return False

    def _process_candle(self, candle: Candle):
        """Process a single candle."""
        current_time = candle.timestamp.timestamp()
        
        # Check if any active orders would be filled
        active_orders_copy = copy.deepcopy(self.active_orders)
        filled_orders = []
        
        for order_id, order_price in active_orders_copy.items():
            if self._would_order_fill(order_id, order_price, candle):
                filled_orders.append(order_id)
                
                # Update trade status
                for trade in self.trades:
                    if trade.order_id == order_id:
                        trade.status = "FILLED"
                        trade.timestamp = candle.timestamp
                        trade.price = candle.close  # Use close price for fill
                        break
                
                if self._is_open_order(order_id):
                    self.filled_orders[order_id] = order_price
                    close_order_id = self._place_close_order(candle.timestamp, order_price, order_id)
                    print(f"[{candle.timestamp}] Open order {order_id} filled at {order_price}, "
                          f"placed close order {close_order_id} at {self.active_orders[close_order_id]}")
                else:
                    # Close order filled, realize profit
                    profit = (self.config.take_profit * self.config.quantity - 
                             2 * self.config.fee_rate * candle.close * self.config.quantity)
                    self.cumulative_pnl += profit
                    print(f"[{candle.timestamp}] Close order {order_id} filled at {order_price}, "
                          f"profit: {profit:.6f}, cumulative PnL: {self.cumulative_pnl:.6f}")
                    
                    # Remove corresponding open order
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

        # Calculate unrealized PnL and update capital
        unrealized = 0
        for open_id, open_price in self.filled_orders.items():
            current_price = candle.close
            direction_factor = 1 if self.config.direction == "BUY" else -1
            unrealized += direction_factor * (current_price - open_price) * self.config.quantity
        current_capital = self.initial_capital + self.cumulative_pnl + unrealized
        self.capital_history.append((candle.timestamp, current_capital))
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        drawdown = (self.peak_capital - current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        if not self.candles:
            return None
            
        start_time = self.candles[0].timestamp
        end_time = self.candles[-1].timestamp
        time_period = (end_time - start_time).total_seconds()
        
        # Convert to days, months, years
        days = time_period / (24 * 3600)
        months = days / 30
        years = days / 365
        
        # Calculate returns using final capital
        final_capital = self.capital_history[-1][1] if self.capital_history else self.initial_capital
        total_return = (final_capital / self.initial_capital - 1) if self.initial_capital > 0 else float('inf')
        annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else float('inf')
        
        return {
            "start_time": start_time,
            "end_time": end_time,
            "period_days": days,
            "period_months": months,
            "period_years": years,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "final_capital": final_capital  # Added missing key
        }
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio based on capital history."""
        if len(self.capital_history) < 2:
            return float('inf')
        returns = pd.Series([c[1] for c in self.capital_history]).pct_change().dropna()
        if len(returns) == 0:
            return float('inf')
        annualized_return = returns.mean() * (252 * 6)  # Assuming 4H candles
        annualized_std = returns.std() * (252 * 6) ** 0.5
        return annualized_return / annualized_std if annualized_std > 0 else float('inf')

    def run_backtest(self):
        """Run the backtest."""
        if not self.candles:
            print("No candle data loaded. Please load data first.")
            return
            
        print(f"Starting backtest with {len(self.candles)} candles")
        print(f"Strategy: {self.config.direction} {self.config.quantity} @{self.config.symbol} "
            f"TP: {self.config.take_profit} Max Orders: {self.config.max_orders}")
        
        # Set initial capital based on max exposure
        first_price = self.candles[0].close
        self.initial_capital = first_price * self.config.quantity * self.config.max_orders
        self.peak_capital = self.initial_capital
        self.capital_history.append((self.candles[0].timestamp, self.initial_capital))
        
        for i, candle in enumerate(self.candles):
            if i % 100 == 0:
                print(f"Processing candle {i}/{len(self.candles)}")
            self._process_candle(candle)
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        if metrics is None:
            print("Error: Failed to calculate metrics")
            return
            
        print("Backtest completed")
        print(f"Total trades: {len(self.trades)}")
        print(f"Initial capital: ${self.initial_capital:.2f} (based on {self.config.quantity} "
            f"contracts * {self.config.max_orders} at price ${first_price:.2f})")
        print(f"Cumulative PnL: {self.cumulative_pnl:.2f}")
        print(f"Final capital: ${metrics['final_capital']:.2f}")  # Fixed to use correct key
        print(f"Time period: {metrics['period_days']:.1f} days ({metrics['period_months']:.1f} months, "
            f"{metrics['period_years']:.2f} years)")
        print(f"Total return: {metrics['total_return']*100:.2f}%")
        print(f"Annualized return: {metrics['annualized_return']*100:.2f}%")
        print(f"Max drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
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
