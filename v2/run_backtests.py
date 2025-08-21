#!/usr/bin/env python3
"""
Simple backtest script to test the trading strategies
"""

import asyncio
import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading_engine import TradingEngine, TradingMode
from strategies import StrategyManager, MovingAverageStrategy, EnhancedMovingAverageStrategy, MeanReversionStrategy, GridTradingStrategy
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class BacktestStrategyConfig:
    """Simple strategy config for backtesting"""
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

async def run_simple_backtest(data_file, symbol, initial_balance=10000):
    """Run a simple backtest"""
    print(f"🔄 Running backtest for {symbol}...")
    print(f"📁 Data file: {data_file}")
    print(f"💰 Initial balance: ${initial_balance}")
    print("=" * 50)
    
    try:
        # Create engine configuration
        engine_config = {
            'exchange_id': 'binance',
            'api_key': 'test',
            'api_secret': 'test',
            'testnet': False,
            'sandbox_mode': False,
            'mode': 'backtest',
            'symbol': symbol,
            'data_file': data_file,
            'initial_balance': initial_balance,
            'base_asset': symbol[:-4] if symbol.endswith('USD') else 'BTC'
        }
        
        # Create trading engine
        engine = TradingEngine(engine_config)
        
        # Initialize engine
        if not await engine.initialize():
            print("❌ Failed to initialize trading engine")
            return None
        
        # Create strategy manager
        strategy_manager = StrategyManager(engine)
        
        # Create and add strategies
        strategies = []
        
        # Enhanced Moving Average strategy
        ma_config = BacktestStrategyConfig(
            name='enhanced_ma_crossover',
            symbol=symbol,
            enabled=True
        )
        ma_strategy = EnhancedMovingAverageStrategy(
            ma_config, engine,
            fast_period=5,
            slow_period=15,
            rsi_period=14,
            volume_period=20
        )
        strategy_manager.add_strategy(ma_strategy)
        strategies.append(ma_strategy)
        
        # Mean reversion strategy
        mr_config = BacktestStrategyConfig(
            name='mean_reversion',
            symbol=symbol,
            enabled=True
        )
        mr_strategy = MeanReversionStrategy(
            mr_config, engine,
            period=10,
            std_dev_threshold=1.5
        )
        strategy_manager.add_strategy(mr_strategy)
        strategies.append(mr_strategy)
        
        # Grid strategy
        grid_config = BacktestStrategyConfig(
            name='grid_trading',
            symbol=symbol,
            enabled=True
        )
        grid_strategy = GridTradingStrategy(
            grid_config, engine,
            grid_spacing=0.005,
            grid_levels=10
        )
        strategy_manager.add_strategy(grid_strategy)
        strategies.append(grid_strategy)
        
        print(f"📊 Strategies loaded: {len(strategies)}")
        
        # Start strategies
        await strategy_manager.start_all_strategies()
        
        # Start engine
        await engine.start()
        
        # Get results
        results = await engine.get_backtest_results()
        
        # Calculate performance metrics
        if results:
            await analyze_results(results, symbol, initial_balance)
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def analyze_results(results, symbol, initial_balance):
    """Analyze backtest results"""
    print("\n📈 Backtest Results Analysis")
    print("=" * 50)
    
    try:
        filled_orders = results.get('filled_orders', [])
        portfolio_values = results.get('portfolio_values', [])
        
        if not filled_orders:
            print("⚠️  No trades executed")
            return
        
        # Basic metrics
        total_trades = len(filled_orders)
        total_fees = results.get('total_fees', 0)
        
        # Calculate profit/loss
        final_value = portfolio_values[-1] if portfolio_values else initial_balance
        total_profit = final_value - initial_balance
        profit_pct = (total_profit / initial_balance) * 100
        
        # Calculate win rate
        buy_trades = [order for order in filled_orders if order.side.value == 'buy']
        sell_trades = [order for order in filled_orders if order.side.value == 'sell']
        
        winning_trades = 0
        for i, sell_order in enumerate(sell_trades):
            # Find corresponding buy order
            for j, buy_order in enumerate(buy_trades):
                if buy_order.created_at < sell_order.created_at:
                    if sell_order.average_price > buy_order.average_price:
                        winning_trades += 1
                    break
        
        win_rate = (winning_trades / len(sell_trades)) * 100 if sell_trades else 0
        
        # Calculate maximum drawdown
        max_drawdown = 0
        peak = portfolio_values[0] if portfolio_values else initial_balance
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe ratio (simplified)
        if len(portfolio_values) > 1:
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = (avg_return / std_return) * np.sqrt(252*24*4) if std_return > 0 else 0  # Annualized for 15m data
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Print results
        print(f"💰 Initial Balance: ${initial_balance:,.2f}")
        print(f"💰 Final Balance: ${final_value:,.2f}")
        print(f"📊 Total Profit/Loss: ${total_profit:,.2f} ({profit_pct:+.2f}%)")
        print(f"📈 Total Trades: {total_trades}")
        print(f"🎯 Win Rate: {win_rate:.1f}%")
        print(f"📉 Max Drawdown: {max_drawdown*100:.2f}%")
        print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"💸 Total Fees: ${total_fees:.2f}")
        
        # Strategy breakdown
        print(f"\n🔍 Strategy Performance:")
        strategy_stats = {}
        for order in filled_orders:
            strategy_name = order.metadata.get('strategy', 'unknown')
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {'trades': 0, 'profit': 0}
            strategy_stats[strategy_name]['trades'] += 1
            # Simplified profit calculation
            if order.side.value == 'sell' and order.average_price > 0:
                strategy_stats[strategy_name]['profit'] += order.filled_quantity * order.average_price * 0.001  # Rough estimate
        
        for strategy, stats in strategy_stats.items():
            print(f"   {strategy}: {stats['trades']} trades, ~${stats['profit']:.2f} profit")
        
        # Risk assessment
        print(f"\n⚠️  Risk Assessment:")
        if max_drawdown > 0.2:
            print("   ❌ High risk - Maximum drawdown exceeded 20%")
        elif max_drawdown > 0.1:
            print("   ⚠️  Moderate risk - Maximum drawdown between 10-20%")
        else:
            print("   ✅ Low risk - Maximum drawdown under 10%")
        
        if profit_pct < 0:
            print("   ❌ Unprofitable strategy")
        elif profit_pct < 5:
            print("   ⚠️  Low profitability - Consider optimization")
        else:
            print("   ✅ Good profitability")
        
        if sharpe_ratio < 1:
            print("   ⚠️  Low risk-adjusted returns")
        elif sharpe_ratio < 2:
            print("   ✅ Moderate risk-adjusted returns")
        else:
            print("   🌟 Excellent risk-adjusted returns")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run backtests on all available data files"""
    print("🧪 Binance Trading Bot v2 - Backtest Suite")
    print("=" * 60)
    
    # Available data files
    data_files = [
        ('processed_data/eth_15m.csv', 'ETHUSD', 'Ethereum 15-minute'),
        ('processed_data/btc_4h.csv', 'BTCUSD', 'Bitcoin 4-hour'),
        ('processed_data/eth_4h.csv', 'ETHUSD', 'Ethereum 4-hour'),
        ('processed_data/doge_4h.csv', 'DOGEUSD', 'Dogecoin 4-hour')
    ]
    
    all_results = {}
    
    for data_file, symbol, description in data_files:
        if os.path.exists(data_file):
            print(f"\n🚀 Testing {description} ({symbol})")
            results = await run_simple_backtest(data_file, symbol)
            all_results[symbol] = results
        else:
            print(f"⚠️  Data file not found: {data_file}")
    
    # Summary
    print(f"\n📊 Backtest Summary")
    print("=" * 60)
    
    profitable_symbols = []
    for symbol, results in all_results.items():
        if results:
            portfolio_values = results.get('portfolio_values', [])
            if portfolio_values:
                final_value = portfolio_values[-1]
                profit = final_value - 10000
                profit_pct = (profit / 10000) * 100
                status = "📈" if profit > 0 else "📉"
                print(f"{status} {symbol}: {profit_pct:+.2f}% (${profit:+,.2f})")
                if profit > 0:
                    profitable_symbols.append(symbol)
    
    print(f"\n🏆 Profitable Symbols: {len(profitable_symbols)}/{len(all_results)}")
    if profitable_symbols:
        print(f"   ✅ {', '.join(profitable_symbols)}")
    
    print(f"\n💡 Recommendations:")
    if len(profitable_symbols) == 0:
        print("   ⚠️  No profitable strategies found. Consider:")
        print("      - Optimizing strategy parameters")
        print("      - Testing different timeframes")
        print("      - Adding risk management features")
    elif len(profitable_symbols) < len(all_results):
        print("   📊 Mixed results. Consider:")
        print("      - Focusing on profitable symbols")
        print("      - Adjusting strategies for unprofitable symbols")
        print("      - Further parameter optimization")
    else:
        print("   🎉 All strategies profitable! Consider:")
        print("      - Fine-tuning for better performance")
        print("      - Testing with live data")
        print("      - Implementing advanced risk management")

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))