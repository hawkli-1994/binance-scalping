import asyncio
import time
import hmac
import hashlib
import aiohttp
import json
from dataclasses import dataclass
from typing import Dict, Optional
from decimal import Decimal
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TradingError(Exception):
    pass

class BinanceAPIError(TradingError):
    pass

@dataclass
class ExchangeInfo:
    symbol: str
    min_qty: float
    max_qty: float
    min_price: float
    max_price: float
    min_notional: float
    price_precision: int
    qty_precision: int
    tick_size: float
    step_size: float

class BinanceClient:
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
        return hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000)

    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False):
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        if params is None:
            params = {}
        if signed:
            params['timestamp'] = self._get_timestamp()
            query_string = "&".join([f"{k}={v}" for k, v in params.items()])
            params['signature'] = self._generate_signature(query_string)
        try:
            async with getattr(self.session, method.lower())(url, headers=headers, params=params if method != 'POST' else None, data=params if method == 'POST' else None) as response:
                data = await response.json()
                if response.status != 200:
                    raise BinanceAPIError(f"API Error {response.status}: {data}")
                return data
        except (aiohttp.ClientError, json.JSONDecodeError) as e:
            raise BinanceAPIError(f"Request failed: {e}")

    async def get_exchange_info(self):
        data = await self._request('GET', '/api/v3/exchangeInfo')
        for symbol_info in data['symbols']:
            if symbol_info['symbol'] == 'BTCUSDT':
                filters = {f['filterType']: f for f in symbol_info['filters']}
                return ExchangeInfo(
                    symbol=symbol_info['symbol'],
                    min_qty=float(filters['LOT_SIZE']['minQty']),
                    max_qty=float(filters['LOT_SIZE']['maxQty']),
                    min_price=float(filters['PRICE_FILTER']['minPrice']),
                    max_price=float(filters['PRICE_FILTER']['maxPrice']),
                    min_notional=float(filters.get('NOTIONAL', {}).get('minNotional', 10)),
                    price_precision=symbol_info['quotePrecision'],
                    qty_precision=symbol_info['baseAssetPrecision'],
                    tick_size=float(filters['PRICE_FILTER']['tickSize']),
                    step_size=float(filters['LOT_SIZE']['stepSize'])
                )
        raise ValueError("Symbol BTCUSDT not found")

    async def get_ticker(self):
        return float((await self._request('GET', '/api/v3/ticker/price', {'symbol': 'BTCUSDT'}))['price'])

    async def get_account(self):
        return await self._request('GET', '/api/v3/account', signed=True)

    async def create_order(self, **params):
        return await self._request('POST', '/api/v3/order', params, signed=True)

    async def get_klines(self, interval: str, limit: int = 20):
        return await self._request('GET', '/api/v3/klines', {'symbol': 'BTCUSDT', 'interval': interval, 'limit': limit})

class TradingBot:
    def __init__(self, config: Dict):
        self.config = self._validate_config(config)
        self.client = BinanceClient(self.config['api_key'], self.config['api_secret'], self.config['testnet'])
        self.exchange_info = None
        self.btc_balance = 0.0
        self.usdt_balance = 0.0
        self.last_price = 0.0
        self.is_running = False
        self.atr = 0.0
        self.active_orders = {}
        self.order_count = 0
        self.daily_trades = 0
        self.last_trade_time = 0

    def _validate_config(self, config: Dict) -> Dict:
        required = {'symbol': str, 'api_key': str, 'api_secret': str, 'target_ratio': float, 
                    'balance_threshold': float, 'min_trade_qty': float, 'atr_period': int, 
                    'stop_loss_atr': float, 'take_profit_atr': float}
        defaults = {'testnet': True, 'trading_fee': 0.001, 'cooldown': 60, 'max_daily_trades': 30}
        validated = config.copy()
        for key, type_ in required.items():
            if key not in config:
                raise ValueError(f"Missing config key: {key}")
            if not isinstance(config[key], type_):
                raise ValueError(f"Invalid type for {key}")
        if not 0 < config['target_ratio'] < 1:
            raise ValueError("target_ratio must be between 0 and 1")
        if not 0 < config['balance_threshold'] < 0.5:
            raise ValueError("balance_threshold must be between 0 and 0.5")
        if config['min_trade_qty'] <= 0:
            raise ValueError("min_trade_qty must be positive")
        validated.update({k: v for k, v in defaults.items() if k not in config})
        return validated

    def _format_qty(self, qty: float) -> str:
        decimal_qty = Decimal(str(qty))
        step_size = Decimal(str(self.exchange_info.step_size))
        return format((decimal_qty // step_size) * step_size, f'.{self.exchange_info.qty_precision}f')

    def _format_price(self, price: float) -> str:
        decimal_price = Decimal(str(price))
        tick_size = Decimal(str(self.exchange_info.tick_size))
        return format((decimal_price // tick_size) * tick_size, f'.{self.exchange_info.price_precision}f')

    def _validate_order(self, qty: float, price: float) -> bool:
        if not (self.exchange_info.min_qty <= qty <= self.exchange_info.max_qty):
            return False
        if not (self.exchange_info.min_price <= price <= self.exchange_info.max_price):
            return False
        if qty * price < self.exchange_info.min_notional:
            return False
        return True

    def _calculate_atr(self, klines: list) -> float:
        if len(klines) < self.config['atr_period']:
            return 0
        trs = []
        for i in range(1, len(klines)):
            high, low, close, prev_close = map(float, (klines[i][2], klines[i][3], klines[i][4], klines[i-1][4]))
            tr = max(high - low, abs(high - prev_close), abs(prev_close - low))
            trs.append(tr)
        return np.mean(trs[-self.config['atr_period']:])

    async def _update_market_data(self):
        try:
            async with self.client:
                self.last_price = await self.client.get_ticker()
                klines = await self.client.get_klines('1m')
                self.atr = self._calculate_atr(klines)
        except Exception as e:
            logger.error(f"Market data update failed: {e}")

    async def _update_balances(self):
        try:
            async with self.client:
                account = await self.client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == 'BTC':
                        self.btc_balance = float(balance['free'])
                    elif balance['asset'] == 'USDT':
                        self.usdt_balance = float(balance['free'])
        except Exception as e:
            logger.error(f"Balance update failed: {e}")

    def _calculate_position_size(self, side: str) -> float:
        portfolio_value = self.btc_balance * self.last_price + self.usdt_balance
        current_ratio = self.btc_balance * self.last_price / portfolio_value if portfolio_value > 0 else 0.5
        target_ratio = self.config['target_ratio']
        threshold = self.config['balance_threshold']

        if side == 'BUY' and current_ratio < target_ratio - threshold:
            needed_btc = (portfolio_value * (target_ratio - current_ratio)) / self.last_price
            max_by_balance = (self.usdt_balance * (1 - self.config['trading_fee'])) / self.last_price
            return max(min(needed_btc, max_by_balance), self.config['min_trade_qty'])
        elif side == 'SELL' and current_ratio > target_ratio + threshold:
            excess_btc = (portfolio_value * (current_ratio - target_ratio)) / self.last_price
            return max(min(excess_btc, self.btc_balance * (1 - self.config['trading_fee'])), self.config['min_trade_qty'])
        return 0

    async def place_order(self, side: str, qty: float, price: float) -> Optional[str]:
        if self.daily_trades >= self.config['max_daily_trades'] or time.time() - self.last_trade_time < self.config['cooldown']:
            return None
        if not self._validate_order(qty, price):
            return None
        try:
            async with self.client:
                order = await self.client.create_order(
                    symbol=self.config['symbol'],
                    side=side,
                    type='LIMIT',
                    quantity=self._format_qty(qty),
                    price=self._format_price(price),
                    timeInForce='GTC',
                    newClientOrderId=f"bot_{int(time.time())}"
                )
                order_id = str(order['orderId'])
                self.active_orders[order_id] = {
                    'side': side,
                    'qty': qty,
                    'price': price,
                    'stop_loss': price * (1 - self.config['stop_loss_atr'] * self.atr / price) if side == 'BUY' else price * (1 + self.config['stop_loss_atr'] * self.atr / price),
                    'take_profit': price * (1 + self.config['take_profit_atr'] * self.atr / price) if side == 'BUY' else price * (1 - self.config['take_profit_atr'] * self.atr / price),
                    'created': time.time()
                }
                self.order_count += 1
                self.daily_trades += 1
                self.last_trade_time = time.time()
                asyncio.create_task(self._monitor_order(order_id))
                return order_id
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return None

    async def _monitor_order(self, order_id: str):
        while order_id in self.active_orders:
            try:
                order = self.active_orders[order_id]
                if time.time() - order['created'] > 300:
                    await self.client.cancel_order(symbol=self.config['symbol'], orderId=order_id)
                    del self.active_orders[order_id]
                    break
                if (order['side'] == 'BUY' and self.last_price <= order['stop_loss']) or \
                   (order['side'] == 'SELL' and self.last_price >= order['stop_loss']):
                    await self.client.create_order(
                        symbol=self.config['symbol'],
                        side='SELL' if order['side'] == 'BUY' else 'BUY',
                        type='MARKET',
                        quantity=self._format_qty(order['qty'])
                    )
                    del self.active_orders[order_id]
                    break
                if (order['side'] == 'BUY' and self.last_price >= order['take_profit']) or \
                   (order['side'] == 'SELL' and self.last_price <= order['take_profit']):
                    await self.client.create_order(
                        symbol=self.config['symbol'],
                        side='SELL' if order['side'] == 'BUY' else 'BUY',
                        type='MARKET',
                        quantity=self._format_qty(order['qty'])
                    )
                    del self.active_orders[order_id]
                    break
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Order monitoring failed: {e}")
                await asyncio.sleep(5)

    async def run(self):
        await self._update_market_data()
        await self._update_balances()
        self.exchange_info = await self.client.get_exchange_info()
        self.is_running = True
        while self.is_running:
            try:
                await self._update_market_data()
                await self._update_balances()
                qty = self._calculate_position_size('BUY')
                if qty > 0:
                    await self.place_order('BUY', qty, self.last_price * 1.001)
                qty = self._calculate_position_size('SELL')
                if qty > 0:
                    await self.place_order('SELL', qty, self.last_price * 0.999)
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(30)

    async def shutdown(self):
        self.is_running = False
        async with self.client:
            for order_id in list(self.active_orders.keys()):
                try:
                    await self.client.cancel_order(symbol=self.config['symbol'], orderId=order_id)
                except Exception as e:
                    logger.error(f"Order cancellation failed: {e}")
        logger.info("Bot shutdown completed")

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    config = {
        'symbol': 'BTCUSDT',
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'testnet': True,
        'target_ratio': 0.5,
        'balance_threshold': 0.03,
        'min_trade_qty': 0.001,
        'atr_period': 14,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.0
    }
    bot = TradingBot(config)
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    asyncio.run(main())