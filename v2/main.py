import asyncio
import json
import uuid
import time
import logging
from datetime import datetime
from typing import Optional, Dict, List
from binance import AsyncClient, BinanceSocketManager
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
config = {
    'symbol': 'BTCUSDT',
    'api_key': 'YOUR_API_KEY',  # 替换为您的Binance API密钥
    'api_secret': 'YOUR_API_SECRET',  # 替换为您的Binance API密钥
    'target_ratio': 0.5,  # 目标BTC/(BTC+USDT)比例
    'balance_threshold': 0.02,  # 再平衡阈值，比例偏离±2%时触发
    'burst_threshold': 2,  # 价格突发阈值（标准差倍数）
    'min_trade_amount': 0.001,  # 最小交易量 (BTC)
    'max_position_ratio': 0.1,  # 单次最大交易占总资产比例
    'poll_interval': 3,  # 轮询间隔（秒）
    'stop_loss': 0.05,  # 止损阈值（5%）
    'volume_lookback': 20,  # 成交量计算回看期
    'price_lookback': 50,  # 价格分析回看期
    'fee_rate': 0.001,  # 交易手续费率
    'max_daily_trades': 100,  # 每日最大交易次数
    'max_drawdown': 0.15,  # 最大回撤限制（15%）
    'emergency_stop': False,  # 紧急停止开关
    'simulation_mode': True,  # 模拟模式开关
}


class RiskManager:
    """风险管理模块"""
    
    def __init__(self, max_drawdown: float = 0.15, max_daily_trades: int = 100):
        self.max_drawdown = max_drawdown
        self.max_daily_trades = max_daily_trades
        self.daily_trades = 0
        self.last_reset_date = datetime.now().date()
        self.peak_value = 0
        self.emergency_stop = False
        
    def reset_daily_counter(self):
        """重置日交易计数器"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.last_reset_date = current_date
            
    def can_trade(self) -> bool:
        """检查是否可以交易"""
        self.reset_daily_counter()
        return (not self.emergency_stop and 
                self.daily_trades < self.max_daily_trades)
    
    def check_drawdown(self, current_value: float) -> bool:
        """检查回撤情况"""
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        if self.peak_value > 0:
            drawdown = (self.peak_value - current_value) / self.peak_value
            if drawdown > self.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%}")
                self.emergency_stop = True
                return False
        return True
    
    def record_trade(self):
        """记录交易"""
        self.daily_trades += 1


class TradingBot:
    def __init__(self):
        self.num_tick = 0
        self.last_trade_id = 0
        self.volume_history = []
        self.order_book = {'asks': [], 'bids': []}
        self.price_history = []
        self.account = None
        self.btc_balance = 0
        self.usdt_balance = 0
        self.portfolio_ratio = 0.5
        self.active_orders = {}
        self.last_price = 0
        self.entry_prices = []
        self.client = None
        self.risk_manager = RiskManager(config['max_drawdown'], config['max_daily_trades'])
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.initial_portfolio_value = 0
        
    async def safe_api_call(self, func, *args, max_retries=3, **kwargs):
        """安全的API调用，包含重试机制"""
        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    def calculate_sma(self, data: List[float], period: int = None) -> float:
        """计算简单移动平均"""
        if not data:
            return 0
        period = period or len(data)
        return np.mean(data[-period:])
    
    def calculate_std(self, data: List[float], period: int = None) -> float:
        """计算标准差"""
        if not data or len(data) < 2:
            return 0
        period = period or len(data)
        return np.std(data[-period:])
    
    def calculate_volume_weighted_price(self) -> float:
        """计算成交量加权平均价格"""
        if len(self.volume_history) < 2:
            return self.last_price
            
        total_volume = sum(vh['volume'] for vh in self.volume_history[-10:])
        if total_volume == 0:
            return self.last_price
            
        vwap = sum(vh['price'] * vh['volume'] for vh in self.volume_history[-10:]) / total_volume
        return vwap
    
    async def update_market_data(self):
        """更新市场数据"""
        try:
            # 更新订单簿
            depth = await self.safe_api_call(
                self.client.get_order_book, 
                symbol=config['symbol'], 
                limit=10
            )
            
            self.order_book = {
                'bids': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['bids']],
                'asks': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['asks']],
            }
            
            if not self.order_book['bids'] or not self.order_book['asks']:
                logger.warning("Empty order book received")
                return
                
            # 计算中间价
            mid_price = (self.order_book['bids'][0]['price'] + self.order_book['asks'][0]['price']) / 2
            
            # 更新价格历史
            if len(self.price_history) >= config['price_lookback']:
                self.price_history.pop(0)
            self.price_history.append(mid_price)
            self.last_price = mid_price
            
            # 更新成交量数据
            trades = await self.safe_api_call(
                self.client.get_recent_trades,
                symbol=config['symbol'],
                limit=config['volume_lookback']
            )
            
            recent_volume = sum(float(trade['qty']) for trade in trades[-5:])  # 最近5笔交易量
            if len(self.volume_history) >= config['volume_lookback']:
                self.volume_history.pop(0)
            self.volume_history.append({
                'price': mid_price,
                'volume': recent_volume,
                'timestamp': time.time()
            })
            
            logger.debug(f"Market data updated - Price: {mid_price:.2f}, Volume: {recent_volume:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def update_account_balance(self):
        """更新账户余额"""
        try:
            account_info = await self.safe_api_call(self.client.get_account)
            self.account = account_info
            
            for balance in account_info['balances']:
                if balance['asset'] == 'BTC':
                    self.btc_balance = float(balance['free'])
                elif balance['asset'] == 'USDT':
                    self.usdt_balance = float(balance['free'])
            
            # 计算投资组合比例
            btc_value = self.btc_balance * self.last_price
            total_value = btc_value + self.usdt_balance
            
            if total_value > 0:
                self.portfolio_ratio = btc_value / total_value
                
                # 设置初始投资组合价值
                if self.initial_portfolio_value == 0:
                    self.initial_portfolio_value = total_value
                    self.risk_manager.peak_value = total_value
                    
                # 检查回撤
                self.risk_manager.check_drawdown(total_value)
            else:
                self.portfolio_ratio = 0.5
                
            logger.debug(f"Account updated - BTC: {self.btc_balance:.6f}, USDT: {self.usdt_balance:.2f}, Ratio: {self.portfolio_ratio:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating account balance: {e}")
    
    async def calculate_trade_amount(self, side: str, target_ratio: float = None) -> float:
        """计算交易数量"""
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        max_trade_value = total_value * config['max_position_ratio']
        
        if side == 'BUY':
            # 买入BTC
            available_usdt = self.usdt_balance * 0.99  # 留一点余量
            max_amount_by_balance = available_usdt / self.last_price
            target_amount = min(max_trade_value / self.last_price, max_amount_by_balance)
            
            if target_ratio:
                # 再平衡计算
                target_btc_value = total_value * target_ratio
                current_btc_value = self.btc_balance * self.last_price
                needed_btc = (target_btc_value - current_btc_value) / self.last_price
                target_amount = min(target_amount, abs(needed_btc) * 0.8)  # 80%执行以避免过度调整
                
        else:  # SELL
            # 卖出BTC
            available_btc = self.btc_balance * 0.99  # 留一点余量
            target_amount = min(max_trade_value / self.last_price, available_btc)
            
            if target_ratio:
                # 再平衡计算
                target_btc_value = total_value * target_ratio
                current_btc_value = self.btc_balance * self.last_price
                needed_reduction = (current_btc_value - target_btc_value) / self.last_price
                target_amount = min(target_amount, abs(needed_reduction) * 0.8)
        
        return max(target_amount, config['min_trade_amount']) if target_amount >= config['min_trade_amount'] else 0
    
    async def place_order(self, side: str, quantity: float, price: float, order_type: str = 'LIMIT') -> Optional[Dict]:
        """下单"""
        if not self.risk_manager.can_trade():
            logger.warning("Trading blocked by risk manager")
            return None
            
        if quantity < config['min_trade_amount']:
            logger.warning(f"Order quantity {quantity:.6f} below minimum {config['min_trade_amount']}")
            return None
            
        try:
            if config['simulation_mode']:
                # 模拟模式
                order_id = str(uuid.uuid4())
                order = {
                    'orderId': order_id,
                    'symbol': config['symbol'],
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'status': 'FILLED',
                    'type': order_type
                }
                
                logger.info(f"[SIMULATION] {side} order: {quantity:.6f} @ {price:.2f}")
                
                # 模拟更新余额
                if side == 'BUY':
                    cost = quantity * price * (1 + config['fee_rate'])
                    if cost <= self.usdt_balance:
                        self.usdt_balance -= cost
                        self.btc_balance += quantity
                        logger.info(f"[SIMULATION] Balance updated - BTC: +{quantity:.6f}, USDT: -{cost:.2f}")
                else:
                    revenue = quantity * price * (1 - config['fee_rate'])
                    if quantity <= self.btc_balance:
                        self.btc_balance -= quantity
                        self.usdt_balance += revenue
                        logger.info(f"[SIMULATION] Balance updated - BTC: -{quantity:.6f}, USDT: +{revenue:.2f}")
            else:
                # 实际交易
                order = await self.safe_api_call(
                    self.client.create_order,
                    symbol=config['symbol'],
                    side=side,
                    type=order_type,
                    quantity=f"{quantity:.6f}",
                    price=f"{price:.2f}",
                    timeInForce='GTC'
                )
                
                logger.info(f"Placed {side} order: {quantity:.6f} @ {price:.2f}, OrderID: {order['orderId']}")
            
            self.active_orders[order['orderId']] = order
            self.risk_manager.record_trade()
            self.total_trades += 1
            
            if side == 'BUY':
                self.entry_prices.append(price)
                
            return order
            
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            self.failed_trades += 1
            return None
    
    async def cancel_old_orders(self):
        """取消旧的未成交订单"""
        if config['simulation_mode']:
            return
            
        try:
            open_orders = await self.safe_api_call(
                self.client.get_open_orders,
                symbol=config['symbol']
            )
            
            for order in open_orders:
                # 取消超过1分钟的订单
                if time.time() - order['time'] / 1000 > 60:
                    await self.safe_api_call(
                        self.client.cancel_order,
                        symbol=config['symbol'],
                        orderId=order['orderId']
                    )
                    logger.info(f"Cancelled old order: {order['orderId']}")
                    
        except Exception as e:
            logger.error(f"Error cancelling old orders: {e}")
    
    async def check_stop_loss(self):
        """检查止损"""
        if not self.entry_prices or self.btc_balance <= config['min_trade_amount']:
            return
            
        avg_entry_price = np.mean(self.entry_prices)
        stop_loss_price = avg_entry_price * (1 - config['stop_loss'])
        
        if self.last_price <= stop_loss_price:
            logger.warning(f"Stop loss triggered! Current: {self.last_price:.2f}, Stop: {stop_loss_price:.2f}")
            
            sell_amount = await self.calculate_trade_amount('SELL')
            if sell_amount > 0:
                await self.place_order('SELL', sell_amount, self.order_book['asks'][0]['price'])
                self.entry_prices = []  # 清空入场价格
    
    async def rebalance_portfolio(self):
        """投资组合再平衡"""
        target_ratio = config['target_ratio']
        threshold = config['balance_threshold']
        
        if abs(self.portfolio_ratio - target_ratio) <= threshold:
            return
            
        logger.info(f"Portfolio rebalancing needed. Current ratio: {self.portfolio_ratio:.3f}, Target: {target_ratio:.3f}")
        
        if self.portfolio_ratio < target_ratio - threshold:
            # BTC比例过低，需要买入BTC
            trade_amount = await self.calculate_trade_amount('BUY', target_ratio)
            if trade_amount > 0:
                await self.place_order('BUY', trade_amount, self.order_book['asks'][0]['price'])
                
        elif self.portfolio_ratio > target_ratio + threshold:
            # BTC比例过高，需要卖出BTC
            trade_amount = await self.calculate_trade_amount('SELL', target_ratio)
            if trade_amount > 0:
                await self.place_order('SELL', trade_amount, self.order_book['bids'][0]['price'])
    
    async def burst_trading_strategy(self):
        """突发价格交易策略"""
        if len(self.price_history) < config['price_lookback']:
            return
            
        sma = self.calculate_sma(self.price_history)
        std = self.calculate_std(self.price_history)
        
        if std == 0:  # 避免除零错误
            return
            
        upper_bound = sma + std * config['burst_threshold']
        lower_bound = sma - std * config['burst_threshold']
        
        # 修正后的交易逻辑：均值回归策略
        if self.last_price > upper_bound:
            # 价格过高，卖出
            trade_amount = await self.calculate_trade_amount('SELL')
            if trade_amount > 0:
                # 使用VWAP调整价格
                vwap = self.calculate_volume_weighted_price()
                adjusted_volume = min(sum(vh['volume'] for vh in self.volume_history[-5:]) / 100, 1.0)
                final_amount = trade_amount * adjusted_volume
                
                if final_amount >= config['min_trade_amount']:
                    await self.place_order('SELL', final_amount, self.order_book['bids'][0]['price'])
                    logger.info(f"Burst sell signal: Price {self.last_price:.2f} > Upper bound {upper_bound:.2f}")
                    
        elif self.last_price < lower_bound:
            # 价格过低，买入
            trade_amount = await self.calculate_trade_amount('BUY')
            if trade_amount > 0:
                # 使用VWAP调整价格
                vwap = self.calculate_volume_weighted_price()
                adjusted_volume = min(sum(vh['volume'] for vh in self.volume_history[-5:]) / 100, 1.0)
                final_amount = trade_amount * adjusted_volume
                
                if final_amount >= config['min_trade_amount']:
                    await self.place_order('BUY', final_amount, self.order_book['asks'][0]['price'])
                    logger.info(f"Burst buy signal: Price {self.last_price:.2f} < Lower bound {lower_bound:.2f}")
    
    def print_status(self):
        """打印状态信息"""
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        pnl = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        status_msg = (
            f"Tick: {self.num_tick} | "
            f"Price: {self.last_price:.2f} | "
            f"Ratio: {self.portfolio_ratio:.3f} | "
            f"Value: {total_value:.2f} | "
            f"PnL: {pnl:+.2f}% | "
            f"Trades: {self.total_trades} | "
            f"Daily: {self.risk_manager.daily_trades}"
        )
        
        logger.info(status_msg)
    
    async def main_loop(self):
        """主循环"""
        self.num_tick += 1
        
        # 检查紧急停止
        if config['emergency_stop'] or self.risk_manager.emergency_stop:
            logger.warning("Emergency stop activated")
            return False
            
        try:
            # 更新市场数据
            await self.update_market_data()
            
            # 更新账户余额
            await self.update_account_balance()
            
            # 取消旧订单
            await self.cancel_old_orders()
            
            # 检查止损
            await self.check_stop_loss()
            
            # 投资组合再平衡（优先级高）
            await self.rebalance_portfolio()
            
            # 突发价格交易策略
            await self.burst_trading_strategy()
            
            # 打印状态
            if self.num_tick % 10 == 0:  # 每10个tick打印一次状态
                self.print_status()
                
            return True
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            return True  # 继续运行，除非是致命错误
    
    async def start(self):
        """启动机器人"""
        logger.info("Starting trading bot...")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        
        # 创建客户端连接
        self.client = await AsyncClient.create(config['api_key'], config['api_secret'])
        
        # 初始化数据
        await self.update_market_data()
        await self.update_account_balance()
        
        logger.info(f"Bot initialized - Initial balance: BTC {self.btc_balance:.6f}, USDT {self.usdt_balance:.2f}")
        
        # 主循环
        while True:
            should_continue = await self.main_loop()
            if not should_continue:
                break
            await asyncio.sleep(config['poll_interval'])
    
    async def close_connection(self):
        """关闭连接"""
        if self.client:
            await self.client.close_connection()
            logger.info("Connection closed")


async def main():
    """主函数"""
    bot = TradingBot()
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.close_connection()


if __name__ == "__main__":
    # 在实际使用前，请确保：
    # 1. 设置正确的API密钥
    # 2. 在模拟模式下充分测试
    # 3. 调整配置参数适合您的风险偏好
    # 4. 理解所有风险
    
    print("⚠️  WARNING: This is a trading bot that can lose money!")
    print("⚠️  Make sure you understand the risks before running!")
    print("⚠️  Start with simulation mode and small amounts!")
    print()
    
    if config['simulation_mode']:
        print("🔧 Running in SIMULATION mode")
    else:
        print("💰 Running in LIVE trading mode")
        
    response = input("Do you want to continue? (yes/no): ")
    if response.lower() in ['yes', 'y']:
        asyncio.run(main())
    else:
        print("Bot cancelled by user")