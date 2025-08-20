import asyncio
import json
import time
import logging
from typing import Dict, Optional
import numpy as np
from binance import AsyncClient

# 导入之前定义的订单执行模块
from order_execution_optimizer import (
    OrderExecutionEngine, OrderType, OrderStatus, 
    SlippageCalculator, SmartPricer
)

logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """集成了高级订单执行优化的交易机器人"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.client = None
        
        # 原有组件
        self.num_tick = 0
        self.price_history = []
        self.volume_history = []
        self.btc_balance = 0
        self.usdt_balance = 0
        self.portfolio_ratio = 0.5
        self.last_price = 0
        self.initial_portfolio_value = 0
        
        # 新增：高级订单执行引擎
        self.execution_engine = None
        
        # 执行质量跟踪
        self.execution_quality_tracker = ExecutionQualityTracker()
        
        # 市场微观结构分析器
        self.market_microstructure = MarketMicrostructureAnalyzer()
        
    async def initialize(self):
        """初始化机器人"""
        logger.info("Initializing Enhanced Trading Bot...")
        
        # 创建Binance客户端
        self.client = await AsyncClient.create(
            self.config['api_key'], 
            self.config['api_secret']
        )
        
        # 初始化订单执行引擎
        execution_config = {
            'simulation_mode': self.config.get('simulation_mode', True),
            'max_chase_attempts': self.config.get('max_chase_attempts', 3),
            'chase_interval': self.config.get('chase_interval', 5.0),
            'max_order_age': self.config.get('max_order_age', 60.0),
            'partial_fill_threshold': self.config.get('partial_fill_threshold', 0.1)
        }
        
        self.execution_engine = OrderExecutionEngine(self.client, execution_config)
        
        # 初始化数据
        await self.update_market_data()
        await self.update_account_balance()
        
        logger.info("Bot initialization completed")
    
    async def update_market_data(self):
        """更新市场数据"""
        try:
            # 获取订单簿
            depth = await self.client.get_order_book(
                symbol=self.config['symbol'], 
                limit=20
            )
            
            order_book = {
                'bids': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['bids']],
                'asks': [{'price': float(price), 'qty': float(qty)} for price, qty in depth['asks']]
            }
            
            # 获取最近交易
            trades = await self.client.get_recent_trades(
                symbol=self.config['symbol'], 
                limit=50
            )
            
            recent_trades = [{
                'price': float(trade['price']),
                'quantity': float(trade['qty']),
                'timestamp': trade['time'],
                'is_buyer_maker': trade['isBuyerMaker']
            } for trade in trades]
            
            # 更新价格历史
            if order_book['bids'] and order_book['asks']:
                mid_price = (order_book['bids'][0]['price'] + order_book['asks'][0]['price']) / 2
                
                if len(self.price_history) >= self.config['price_lookback']:
                    self.price_history.pop(0)
                self.price_history.append(mid_price)
                self.last_price = mid_price
            
            # 更新智能定价器的市场数据
            self.execution_engine.smart_pricer.update_market_data(order_book, recent_trades)
            
            # 更新市场微观结构分析
            self.market_microstructure.update(order_book, recent_trades)
            
            logger.debug(f"Market data updated - Price: {mid_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def update_account_balance(self):
        """更新账户余额"""
        try:
            account_info = await self.client.get_account()
            
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
                if self.initial_portfolio_value == 0:
                    self.initial_portfolio_value = total_value
            
            logger.debug(f"Account updated - BTC: {self.btc_balance:.6f}, USDT: {self.usdt_balance:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating account balance: {e}")
    
    async def execute_smart_trade(self, side: str, target_amount: float, urgency: float = 0.5, 
                                 strategy_name: str = "default") -> Optional[str]:
        """执行智能交易"""
        if target_amount < self.config['min_trade_amount']:
            logger.warning(f"Trade amount {target_amount:.6f} below minimum")
            return None
        
        # 获取市场条件评估
        market_condition = self.market_microstructure.get_market_condition()
        
        # 根据市场条件调整执行策略
        if market_condition['volatility'] == 'HIGH':
            # 高波动环境：使用更保守的策略
            urgency *= 0.7
            max_slippage = 0.002  # 0.2%
        elif market_condition['liquidity'] == 'LOW':
            # 低流动性环境：更加谨慎
            urgency *= 0.5
            max_slippage = 0.001  # 0.1%
        else:
            max_slippage = 0.003  # 0.3%
        
        # 智能拆分大订单
        if target_amount > self.config.get('large_order_threshold', 0.01):
            return await self._execute_large_order(side, target_amount, urgency, max_slippage)
        else:
            return await self._execute_single_order(side, target_amount, urgency, max_slippage, strategy_name)
    
    async def _execute_single_order(self, side: str, amount: float, urgency: float, 
                                   max_slippage: float, strategy_name: str) -> Optional[str]:
        """执行单个订单"""
        # 根据紧急度选择订单类型
        if urgency > 0.9:
            order_type = OrderType.MARKET
        else:
            order_type = OrderType.LIMIT
        
        # 下单
        order = await self.execution_engine.place_order(
            symbol=self.config['symbol'],
            side=side,
            quantity=amount,
            order_type=order_type,
            urgency=urgency,
            max_slippage=max_slippage
        )
        
        if order:
            # 记录执行质量
            self.execution_quality_tracker.record_order_start(
                order.order_id, strategy_name, urgency, amount
            )
            logger.info(f"Smart trade executed: {side} {amount:.6f} BTC, Order ID: {order.order_id}")
            return order.order_id
        
        return None
    
    async def _execute_large_order(self, side: str, total_amount: float, urgency: float, 
                                  max_slippage: float) -> Optional[str]:
        """执行大额订单（智能拆分）"""
        # 计算拆分策略
        num_slices = min(5, max(2, int(total_amount / self.config['min_trade_amount'] / 10)))
        slice_amount = total_amount / num_slices
        
        logger.info(f"Large order execution: splitting {total_amount:.6f} into {num_slices} slices")
        
        executed_orders = []
        total_executed = 0
        
        for i in range(num_slices):
            # 计算当前切片的数量（最后一片可能略有不同）
            if i == num_slices - 1:
                current_amount = total_amount - total_executed
            else:
                current_amount = slice_amount
            
            # 根据市场条件调整执行间隔
            if i > 0:
                interval = self._calculate_execution_interval()
                await asyncio.sleep(interval)
            
            # 执行切片
            order_id = await self._execute_single_order(
                side, current_amount, urgency, max_slippage, "large_order_slice"
            )
            
            if order_id:
                executed_orders.append(order_id)
                total_executed += current_amount
            else:
                logger.warning(f"Failed to execute slice {i+1}/{num_slices}")
                break
        
        return f"large_order_{len(executed_orders)}_slices" if executed_orders else None
    
    def _calculate_execution_interval(self) -> float:
        """计算执行间隔"""
        # 根据市场微观结构调整间隔
        market_condition = self.market_microstructure.get_market_condition()
        
        base_interval = 2.0  # 基础间隔2秒
        
        if market_condition['volatility'] == 'HIGH':
            return base_interval * 0.5  # 高波动时快速执行
        elif market_condition['liquidity'] == 'LOW':
            return base_interval * 2.0  # 低流动性时放慢执行
        else:
            return base_interval
    
    async def rebalance_portfolio_enhanced(self):
        """增强版投资组合再平衡"""
        target_ratio = self.config['target_ratio']
        threshold = self.config['balance_threshold']
        
        if abs(self.portfolio_ratio - target_ratio) <= threshold:
            return
        
        # 计算需要交易的数量
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        target_btc_value = total_value * target_ratio
        current_btc_value = self.btc_balance * self.last_price
        
        if self.portfolio_ratio < target_ratio - threshold:
            # 需要买入BTC
            needed_btc = (target_btc_value - current_btc_value) / self.last_price
            max_buy = self.usdt_balance * 0.99 / self.last_price
            trade_amount = min(needed_btc * 0.8, max_buy)  # 80%执行，留出缓冲
            
            if trade_amount >= self.config['min_trade_amount']:
                logger.info(f"Rebalancing: buying {trade_amount:.6f} BTC (ratio {self.portfolio_ratio:.3f} -> {target_ratio:.3f})")
                await self.execute_smart_trade('BUY', trade_amount, urgency=0.3, strategy_name="rebalance")
        
        elif self.portfolio_ratio > target_ratio + threshold:
            # 需要卖出BTC
            needed_reduction = (current_btc_value - target_btc_value) / self.last_price
            max_sell = self.btc_balance * 0.99
            trade_amount = min(needed_reduction * 0.8, max_sell)
            
            if trade_amount >= self.config['min_trade_amount']:
                logger.info(f"Rebalancing: selling {trade_amount:.6f} BTC (ratio {self.portfolio_ratio:.3f} -> {target_ratio:.3f})")
                await self.execute_smart_trade('SELL', trade_amount, urgency=0.3, strategy_name="rebalance")
    
    async def burst_trading_strategy_enhanced(self):
        """增强版突发价格交易策略"""
        if len(self.price_history) < self.config['price_lookback']:
            return
        
        # 计算技术指标
        sma = np.mean(self.price_history)
        std = np.std(self.price_history)
        
        if std == 0:
            return
        
        # 获取市场条件
        market_condition = self.market_microstructure.get_market_condition()
        
        # 根据市场条件调整阈值
        base_threshold = self.config['burst_threshold']
        if market_condition['volatility'] == 'HIGH':
            threshold = base_threshold * 1.5  # 提高阈值避免假信号
        elif market_condition['trend_strength'] > 0.7:
            threshold = base_threshold * 0.7  # 降低阈值捕捉趋势
        else:
            threshold = base_threshold
        
        upper_bound = sma + std * threshold
        lower_bound = sma - std * threshold
        
        # 计算交易量
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        max_trade_value = total_value * self.config['max_position_ratio']
        
        if self.last_price > upper_bound and self.btc_balance > self.config['min_trade_amount']:
            # 价格过高，卖出
            trade_amount = min(
                max_trade_value / self.last_price,
                self.btc_balance * 0.3  # 最多卖出30%的BTC
            )
            
            if trade_amount >= self.config['min_trade_amount']:
                urgency = min(0.8, (self.last_price - upper_bound) / (std * threshold))
                logger.info(f"Burst sell signal: price {self.last_price:.2f} > upper bound {upper_bound:.2f}")
                await self.execute_smart_trade('SELL', trade_amount, urgency=urgency, strategy_name="burst_sell")
        
        elif self.last_price < lower_bound and self.usdt_balance > 10:  # 至少10 USDT
            # 价格过低，买入
            trade_amount = min(
                max_trade_value / self.last_price,
                self.usdt_balance * 0.3 / self.last_price  # 最多用30%的USDT
            )
            
            if trade_amount >= self.config['min_trade_amount']:
                urgency = min(0.8, (lower_bound - self.last_price) / (std * threshold))
                logger.info(f"Burst buy signal: price {self.last_price:.2f} < lower bound {lower_bound:.2f}")
                await self.execute_smart_trade('BUY', trade_amount, urgency=urgency, strategy_name="burst_buy")
    
    async def main_loop(self):
        """主循环"""
        self.num_tick += 1
        
        try:
            # 更新市场数据
            await self.update_market_data()
            
            # 更新账户余额
            await self.update_account_balance()
            
            # 执行交易策略
            await self.rebalance_portfolio_enhanced()
            await self.burst_trading_strategy_enhanced()
            
            # 定期打印状态和执行质量报告
            if self.num_tick % 20 == 0:
                await self.print_comprehensive_status()
            
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
    
    async def print_comprehensive_status(self):
        """打印综合状态"""
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        pnl = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        # 基础状态
        logger.info(f"=== Status Report (Tick {self.num_tick}) ===")
        logger.info(f"Price: {self.last_price:.2f} | Ratio: {self.portfolio_ratio:.3f} | Value: {total_value:.2f} | PnL: {pnl:+.2f}%")
        
        # 订单执行统计
        exec_stats = self.execution_engine.get_execution_statistics()
        if exec_stats:
            logger.info(f"Execution: {exec_stats['total_orders']} orders, {exec_stats['average_fill_rate']:.1%} fill rate, {exec_stats['average_slippage']:.4f} avg slippage")
        
        # 活跃订单
        active_orders = self.execution_engine.get_active_orders_summary()
        if active_orders['count'] > 0:
            logger.info(f"Active Orders: {active_orders['count']} orders, avg age: {active_orders['average_age']:.1f}s")
        
        # 市场条件
        market_condition = self.market_microstructure.get_market_condition()
        logger.info(f"Market: {market_condition['volatility']} volatility, {market_condition['liquidity']} liquidity")
        
        # 执行质量
        quality_report = self.execution_quality_tracker.get_quality_report()
        if quality_report:
            logger.info(f"Quality: {quality_report['success_rate']:.1%} success, {quality_report['avg_execution_time']:.1f}s avg time")
    
    async def shutdown(self):
        """优雅关闭"""
        logger.info("Shutting down Enhanced Trading Bot...")
        
        # 取消所有活跃订单
        cancelled_count = await self.execution_engine.cancel_all_orders()
        if cancelled_count > 0:
            logger.info(f"Cancelled {cancelled_count} active orders")
        
        # 生成最终报告
        await self.generate_final_report()
        
        # 关闭连接
        if self.client:
            await self.client.close_connection()
        
        logger.info("Bot shutdown completed")
    
    async def generate_final_report(self):
        """生成最终报告"""
        total_value = self.btc_balance * self.last_price + self.usdt_balance
        total_return = ((total_value - self.initial_portfolio_value) / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0
        
        exec_stats = self.execution_engine.get_execution_statistics()
        quality_report = self.execution_quality_tracker.get_quality_report()
        
        report = {
            'summary': {
                'initial_value': self.initial_portfolio_value,
                'final_value': total_value,
                'total_return_pct': total_return,
                'runtime_ticks': self.num_tick
            },
            'execution_performance': exec_stats,
            'quality_metrics': quality_report
        }
        
        logger.info("=== FINAL PERFORMANCE REPORT ===")
        logger.info(json.dumps(report, indent=2))


class ExecutionQualityTracker:
    """执行质量跟踪器"""
    
    def __init__(self):
        self.order_records = {}
        self.completed_orders = []
    
    def record_order_start(self, order_id: str, strategy: str, urgency: float, amount: float):
        """记录订单开始"""
        self.order_records[order_id] = {
            'strategy': strategy,
            'urgency': urgency,
            'amount': amount,
            'start_time': time.time()
        }
    
    def record_order_completion(self, order_id: str, fill_rate: float, avg_price: float, status: str):
        """记录订单完成"""
        if order_id in self.order_records:
            record = self.order_records.pop(order_id)
            record.update({
                'fill_rate': fill_rate,
                'avg_price': avg_price,
                'status': status,
                'execution_time': time.time() - record['start_time']
            })
            self.completed_orders.append(record)
    
    def get_quality_report(self) -> Dict:
        """获取质量报告"""
        if not self.completed_orders:
            return {}
        
        successful_orders = [o for o in self.completed_orders if o['status'] == 'FILLED']
        
        return {
            'total_orders': len(self.completed_orders),
            'successful_orders': len(successful_orders),
            'success_rate': len(successful_orders) / len(self.completed_orders),
            'avg_execution_time': np.mean([o['execution_time'] for o in self.completed_orders]),
            'avg_fill_rate': np.mean([o['fill_rate'] for o in self.completed_orders])
        }


class MarketMicrostructureAnalyzer:
    """市场微观结构分析器"""
    
    def __init__(self):
        self.order_book_history = []
        self.trade_history = []
        self.max_history = 100
    
    def update(self, order_book: Dict, trades: List[Dict]):
        """更新市场数据"""
        # 记录订单簿
        if len(self.order_book_history) >= self.max_history:
            self.order_book_history.pop(0)
        
        self.order_book_history.append({
            'timestamp': time.time(),
            'bid': order_book['bids'][0]['price'] if order_book['bids'] else 0,
            'ask': order_book['asks'][0]['price'] if order_book['asks'] else 0,
            'bid_size': order_book['bids'][0]['qty'] if order_book['bids'] else 0,
            'ask_size': order_book['asks'][0]['qty'] if order_book['asks'] else 0,
            'spread': (order_book['asks'][0]['price'] - order_book['bids'][0]['price']) if (order_book['bids'] and order_book['asks']) else 0
        })
        
        # 记录交易
        self.trade_history.extend(trades[-10:])  # 只保留最近10笔
        if len(self.trade_history) > self.max_history:
            self.trade_history = self.trade_history[-self.max_history:]
    
    def get_market_condition(self) -> Dict:
        """获取市场条件"""
        if len(self.order_book_history) < 10:
            return {'volatility': 'UNKNOWN', 'liquidity': 'UNKNOWN', 'trend_strength': 0}
        
        recent_spreads = [ob['spread'] for ob in self.order_book_history[-20:]]
        recent_sizes = [(ob['bid_size'] + ob['ask_size']) for ob in self.order_book_history[-20:]]
        
        # 波动率评估
        avg_spread = np.mean(recent_spreads)
        spread_std = np.std(recent_spreads)
        volatility = 'HIGH' if spread_std > avg_spread * 0.1 else 'NORMAL' if spread_std > avg_spread * 0.05 else 'LOW'
        
        # 流动性评估
        avg_size = np.mean(recent_sizes)
        liquidity = 'HIGH' if avg_size > 1.0 else 'NORMAL' if avg_size > 0.5 else 'LOW'
        
        # 趋势强度（简化版）
        if len(self.order_book_history) >= 20:
            prices = [(ob['bid'] + ob['ask']) / 2 for ob in self.order_book_history[-20:]]
            trend_strength = abs(np.corrcoef(range(len(prices)), prices)[0, 1]) if len(prices) > 1 else 0
        else:
            trend_strength = 0
        
        return {
            'volatility': volatility,
            'liquidity': liquidity,
            'trend_strength': trend_strength,
            'avg_spread': avg_spread,
            'avg_size': avg_size
        }


# 使用示例
async def main():
    config = {
        'symbol': 'BTCUSDT',
        'api_key': 'YOUR_API_KEY',
        'api_secret': 'YOUR_API_SECRET',
        'target_ratio': 0.5,
        'balance_threshold': 0.02,
        'burst_threshold': 2,
        'min_trade_amount': 0.001,
        'max_position_ratio': 0.1,
        'price_lookback': 50,
        'simulation_mode': True,
        'large_order_threshold': 0.01,
        'max_chase_attempts': 3,
        'chase_interval': 5.0,
        'max_order_age': 60.0,
        'partial_fill_threshold': 0.1
    }
    
    bot = EnhancedTradingBot(config)
    
    try:
        await bot.initialize()
        
        # 主循环
        while True:
            await bot.main_loop()
            await asyncio.sleep(3)  # 3秒轮询间隔
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())