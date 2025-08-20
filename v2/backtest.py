import pandas as pd
import numpy as np
import uuid
from datetime import datetime

# Configuration
config = {
    'symbol': 'ETHUSDT',
    'initial_usdt': 10000,  # Initial USDT balance
    'initial_eth': 0,  # Initial ETH balance
    'target_ratio': 0.5,  # Target ETH/(ETH+USDT) ratio
    'balance_threshold': 0.02,  # Rebalance if ratio deviates by ±2%
    'burst_threshold': 2,  # Price movement threshold (in standard deviations)
    'min_trade_amount': 0.001,  # Minimum trade size (ETH)
    'max_position': 0.1,  # Maximum position size (ETH)
    'stop_loss': 0.05,  # Stop-loss threshold (5% below entry price)
    'volume_lookback': 10,  # Number of trades for volume calculation
    'price_lookback': 20,  # Number of prices for burst detection
    'fee_rate': 0.001,  # 0.1% maker fee
    'column_mapping': {  # Map expected column names to CSV MultiIndex columns
        'datetime': 'Datetime',  # Index column
        'price': ('Price', 'ETH-USD'),
        'close': ('Close', 'ETH-USD'),
        'high': ('High', 'ETH-USD'),
        'low': ('Low', 'ETH-USD'),
        'volume': ('Volume', 'ETH-USD')
    }
}

class BacktestBot:
    def __init__(self):
        self.num_tick = 0
        self.volume = 0
        self.prices = []
        self.account = {'eth': config['initial_eth'], 'usdt': config['initial_usdt']}
        self.eth = config['initial_eth']  # ETH balance
        self.usdt = config['initial_usdt']
        self.p = 0.5
        self.active_order_id = None
        self.entry_price = 0
        self.last_price = 0
        self.trades = []
        self.net_values = []
        self.current_timestamp = None

    def calculate_sma(self, data):
        return np.mean(data)

    def calculate_sd(self, data, mean):
        return np.std(data)

    def simulate_order_book(self, high, low):
        return {
            'bids': [{'price': low, 'qty': 1}],  # Approximate bid as Low price
            'asks': [{'price': high, 'qty': 1}]  # Approximate ask as High price
        }

    def update_trades(self, volume):
        self.volume = 0.7 * self.volume + 0.3 * float(volume)
        if len(self.prices) == 0:
            self.prices = [self.last_price] * config['price_lookback']

    def update_order_book(self, close, high, low):
        self.order_book = self.simulate_order_book(high, low)
        if len(self.prices) >= config['price_lookback']:
            self.prices.pop(0)
        self.prices.append(float(close))
        self.last_price = float(close)

    def balance_account(self):
        eth_value = self.eth * self.last_price
        self.p = eth_value / (eth_value + self.usdt) if (eth_value + self.usdt) > 0 else 0.5

        if self.p < config['target_ratio'] - config['balance_threshold']:
            print(f"Rebalancing: Portfolio ratio {self.p:.3f} too low")
            buy_amount = min((self.usdt * 0.01) / self.last_price, config['max_position'])
            if buy_amount >= config['min_trade_amount']:
                self.place_order('BUY', buy_amount, self.order_book['bids'][0]['price'])

        elif self.p > config['target_ratio'] + config['balance_threshold']:
            print(f"Rebalancing: Portfolio ratio {self.p:.3f} too high")
            sell_amount = min(self.eth * 0.01, config['max_position'])
            if sell_amount >= config['min_trade_amount']:
                self.place_order('SELL', sell_amount, self.order_book['asks'][0]['price'])

    def place_order(self, side, quantity, price):
        order_id = str(uuid.uuid4())
        self.active_order_id = order_id
        executed_price = price  # Use provided price for execution
        deal_amount = quantity
        fee = quantity * executed_price * config['fee_rate']

        if side == 'BUY':
            cost = quantity * executed_price + fee
            if self.usdt >= cost:
                self.eth += deal_amount
                self.usdt -= cost
                self.entry_price = executed_price
                self.trades.append({
                    'side': side,
                    'quantity': quantity,
                    'price': executed_price,
                    'fee': fee,
                    'timestamp': self.current_timestamp
                })
                print(f"Executed BUY: {quantity:.8f} ETH @ {executed_price:.2f}")
            else:
                deal_amount = 0
        elif side == 'SELL':
            if self.eth >= quantity:
                self.eth -= quantity
                self.usdt += quantity * executed_price - fee
                self.trades.append({
                    'side': side,
                    'quantity': quantity,
                    'price': executed_price,
                    'fee': fee,
                    'timestamp': self.current_timestamp
                })
                print(f"Executed SELL: {quantity:.8f} ETH @ {executed_price:.2f}")
            else:
                deal_amount = 0

        self.active_order_id = None
        status = 'FILLED' if deal_amount > 0 else 'CANCELED'
        return {'order_id': order_id, 'status': status, 'executed_qty': deal_amount}

    def check_stop_loss(self):
        if self.entry_price and self.last_price and self.eth > 0:
            if self.last_price < self.entry_price * (1 - config['stop_loss']):
                print('Stop-loss triggered')
                self.place_order('SELL', min(self.eth, config['max_position']), self.order_book['asks'][0]['price'])
                self.entry_price = 0

    def process_row(self, row, index):
        self.num_tick += 1
        column_map = config['column_mapping']
        self.current_timestamp = index  # Use DataFrame index as timestamp
        self.update_trades(row[column_map['volume']])
        self.update_order_book(row[column_map['close']], row[column_map['high']], row[column_map['low']])
        self.balance_account()
        self.check_stop_loss()

        if len(self.prices) < config['price_lookback']:
            return

        sma = self.calculate_sma(self.prices)
        sd = self.calculate_sd(self.prices, sma)
        burst_price = sd * config['burst_threshold']
        trade_amount = 0
        side = None

        if self.last_price > sma + burst_price:
            side = 'BUY'
            trade_amount = min((self.usdt * 0.5) / self.last_price, config['max_position'])
        elif self.last_price < sma - burst_price:
            side = 'SELL'
            trade_amount = min(self.eth * 0.5, config['max_position'])

        trade_amount *= min(self.volume / 1000, 1)
        if self.num_tick < 10:
            trade_amount *= 0.8

        if side and trade_amount >= config['min_trade_amount']:
            price = self.order_book['bids'][0]['price'] if side == 'BUY' else self.order_book['asks'][0]['price']
            order = self.place_order(side, trade_amount, price)
            if order['status'] == 'FILLED':
                remaining = trade_amount - order['executed_qty']
                while remaining >= config['min_trade_amount']:
                    remaining *= 0.9
                    if remaining >= config['min_trade_amount']:
                        self.place_order(side, remaining, price)

        # Track net value
        net_value = self.usdt + self.eth * self.last_price
        self.net_values.append(net_value)
        print(f"Tick: {self.num_tick}, Price: {self.last_price:.2f}, Ratio: {self.p:.3f}, Net Value: {net_value:.2f}")

    def calculate_metrics(self):
        if len(self.prices) == 0:
            return {
                'initial_value': 0,
                'final_value': 0,
                'profit': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0
            }
        initial_value = config['initial_usdt'] + config['initial_eth'] * self.prices[0]
        final_value = self.usdt + self.eth * self.last_price
        profit = final_value - initial_value
        total_trades = len(self.trades)
        win_trades = 0
        for i, trade in enumerate(self.trades):
            if trade['side'] == 'SELL':
                # Find previous buy price
                for j in range(i-1, -1, -1):
                    if self.trades[j]['side'] == 'BUY':
                        if trade['price'] > self.trades[j]['price']:
                            win_trades += 1
                        break

        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate maximum drawdown
        max_drawdown = 0
        peak = self.net_values[0] if self.net_values else 0
        for value in self.net_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return {
            'initial_value': round(initial_value, 2),
            'final_value': round(final_value, 2),
            'profit': round(profit, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'max_drawdown': round(max_drawdown * 100, 2)
        }

    def run_backtest(self, csv_file_path):
        print('Starting backtest...')
        # Read CSV with multi-level header and set Datetime as index
        df = pd.read_csv(csv_file_path, header=[0, 1], index_col=0)
        for index, row in df.iterrows():
            self.process_row(row, index)

        metrics = self.calculate_metrics()
        print('\nBacktest Results:')
        print(f"Initial Portfolio Value: {metrics['initial_value']} USDT")
        print(f"Final Portfolio Value: {metrics['final_value']} USDT")
        print(f"Profit/Loss: {metrics['profit']} USDT")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']}%")
        print(f"Maximum Drawdown: {metrics['max_drawdown']}%")

# Main function
if __name__ == "__main__":
    bot = BacktestBot()
    bot.run_backtest('/home/krli/dreams/binance-scalping/data/yahoo_eth_usd_15m.csv')