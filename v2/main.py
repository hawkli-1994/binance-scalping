import asyncio
import json
import uuid
from binance import AsyncClient, BinanceSocketManager
import numpy as np

# Configuration
config = {
    'symbol': 'BTCUSDT',
    'api_key': 'YOUR_API_KEY',  # Replace with your Binance API key
    'api_secret': 'YOUR_API_SECRET',  # Replace with your Binance API secret
    'target_ratio': 0.5,  # Target BTC/(BTC+USDT) ratio
    'balance_threshold': 0.02,  # Rebalance if ratio deviates by ±2%
    'burst_threshold': 2,  # Price movement threshold (in standard deviations)
    'min_trade_amount': 0.001,  # Minimum trade size (BTC)
    'max_position': 0.1,  # Maximum position size (BTC)
    'poll_interval': 5,  # Polling interval (seconds)
    'stop_loss': 0.05,  # Stop-loss threshold (5% below entry price)
    'volume_lookback': 10,  # Number of trades for volume calculation
    'price_lookback': 20,  # Number of prices for burst detection
    'fee_rate': 0.001,  # 0.1% maker fee
}


class TradingBot:
    def __init__(self):
        self.num_tick = 0
        self.last_trade_id = 0
        self.volume = 0
        self.order_book = {'asks': [], 'bids': []}
        self.prices = []
        self.account = None
        self.btc = 0
        self.usdt = 0
        self.p = 0.5
        self.active_order_id = None
        self.last_price = 0
        self.entry_price = 0
        self.client = None

    def calculate_sma(self, data):
        """Calculate simple moving average"""
        return np.mean(data)

    def calculate_sd(self, data, mean):
        """Calculate standard deviation"""
        return np.std(data)

    async def update_trades(self):
        """Update trade volume"""
        try:
            trades = await self.client.get_recent_trades(symbol=config['symbol'], limit=config['volume_lookback'])
            total_volume = 0
            for trade in trades:
                trade_id = int(trade['id'])
                if trade_id > self.last_trade_id:
                    self.last_trade_id = trade_id
                    total_volume += float(trade['qty'])
            self.volume = 0.7 * self.volume + 0.3 * total_volume
            if len(self.prices) == 0:
                self.prices = [float(trades[0]['price'])] * config['price_lookback']
        except Exception as e:
            print(f"Error fetching trades: {e}")

    async def update_order_book(self):
        """Update order book and prices"""
        try:
            depth = await self.client.get_order_book(symbol=config['symbol'], limit=5)
            self.order_book = {
                'bids': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['bids']],
                'asks': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['asks']],
            }
            if len(self.order_book['bids']) < 3 or len(self.order_book['asks']) < 3:
                return
            
            mid_price = (self.order_book['bids'][0]['price'] + self.order_book['asks'][0]['price']) / 2
            
            if len(self.prices) >= config['price_lookback']:
                self.prices.pop(0)
            self.prices.append(mid_price)
            self.last_price = mid_price
        except Exception as e:
            print(f"Error fetching order book: {e}")

    async def balance_account(self):
        """Update account balance and portfolio ratio"""
        try:
            account_info = await self.client.get_account()
            self.account = account_info
            for balance in account_info['balances']:
                if balance['asset'] == 'BTC':
                    self.btc = float(balance['free'])
                elif balance['asset'] == 'USDT':
                    self.usdt = float(balance['free'])
            
            btc_value = self.btc * self.last_price
            self.p = btc_value / (btc_value + self.usdt) if (btc_value + self.usdt) > 0 else 0.5

            if self.p < config['target_ratio'] - config['balance_threshold']:
                print(f"Rebalancing: Portfolio ratio {self.p:.3f} too low")
                buy_amount = min((self.usdt * 0.01) / self.last_price, config['max_position'])
                if buy_amount >= config['min_trade_amount']:
                    await self.place_order('BUY', buy_amount, self.order_book['bids'][0]['price'])
            
            elif self.p > config['target_ratio'] + config['balance_threshold']:
                print(f"Rebalancing: Portfolio ratio {self.p:.3f} too high")
                sell_amount = min(self.btc * 0.01, config['max_position'])
                if sell_amount >= config['min_trade_amount']:
                    await self.place_order('SELL', sell_amount, self.order_book['asks'][0]['price'])

            # Cancel old orders
            open_orders = await self.client.get_open_orders(symbol=config['symbol'])
            for order in open_orders:
                if order['orderId'] != self.active_order_id:
                    await self.client.cancel_order(symbol=config['symbol'], orderId=order['orderId'])
        except Exception as e:
            print(f"Error fetching account: {e}")

    async def place_order(self, side, quantity, price):
        """Place an order"""
        try:
            order = await self.client.create_order(
                symbol=config['symbol'],
                side=side,
                type='LIMIT',
                quantity=f"{quantity:.8f}",
                price=f"{price:.2f}",
                timeInForce='GTC'
            )
            self.active_order_id = order['orderId']
            if side == 'BUY':
                self.entry_price = price
            print(f"Placed {side} order: {quantity:.8f} @ {price:.2f}")
            return order
        except Exception as e:
            print(f"Error placing {side} order: {e}")
            return None

    async def check_stop_loss(self):
        """Check stop-loss"""
        if self.entry_price and self.last_price and self.btc > 0:
            if self.last_price < self.entry_price * (1 - config['stop_loss']):
                print('Stop-loss triggered')
                await self.place_order('SELL', min(self.btc, config['max_position']), self.order_book['asks'][0]['price'])
                self.entry_price = 0  # Reset entry price

    async def poll(self):
        """Main polling loop"""
        self.num_tick += 1
        await self.update_trades()
        await self.update_order_book()
        await self.balance_account()
        await self.check_stop_loss()

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
            trade_amount = min(self.btc * 0.5, config['max_position'])

        # Adjust trade amount based on volume
        trade_amount *= min(self.volume / 1000, 1)  # Scale by volume (arbitrary 1000 as reference)
        if self.num_tick < 10:
            trade_amount *= 0.8  # Reduce size for initial ticks

        if side and trade_amount >= config['min_trade_amount']:
            price = self.order_book['bids'][0]['price'] if side == 'BUY' else self.order_book['asks'][0]['price']
            order = await self.place_order(side, trade_amount, price)
            if order:
                remaining = trade_amount
                while remaining >= config['min_trade_amount']:
                    order_status = await self.client.get_order(symbol=config['symbol'], orderId=order['orderId'])
                    if order_status['status'] == 'FILLED':
                        break
                    elif order_status['status'] == 'PARTIALLY_FILLED':
                        executed_qty = float(order_status['executedQty'])
                        remaining -= executed_qty
                    else:
                        await self.client.cancel_order(symbol=config['symbol'], orderId=order['orderId'])
                        break
                    await asyncio.sleep(0.2)
                self.active_order_id = None

        print(f"Tick: {self.num_tick}, Price: {self.last_price:.2f}, Ratio: {self.p:.3f}, Volume: {self.volume:.2f}")

    async def start(self):
        """Start the bot"""
        print('Starting trading bot...')
        self.client = await AsyncClient.create(config['api_key'], config['api_secret'])
        while True:
            await self.poll()
            await asyncio.sleep(config['poll_interval'])

    async def close_connection(self):
        """Close the connection"""
        if self.client:
            await self.client.close_connection()


async def main():
    bot = TradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("Stopping bot...")
    finally:
        await bot.close_connection()


if __name__ == "__main__":
    asyncio.run(main())