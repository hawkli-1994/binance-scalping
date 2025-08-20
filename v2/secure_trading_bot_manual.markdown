# 安全交易机器人使用说明书（v4.0）

## 概述

本安全交易机器人（Secure Trading Bot v4.0）是一款专为加密货币交易设计的自动化交易工具，基于Binance交易所API，支持BTC/USDT交易对。它采用增强的再平衡策略，结合ATR（平均真实波幅）止损和止盈机制，适合低风险、稳健的交易场景。机器人支持实时交易、模拟交易以及回测模式，帮助用户优化策略并验证性能。

---

## 功能特点

1. **增强再平衡策略**：
   - 维持目标投资组合比例（BTC/USDT），通过低买高卖实现潜在盈利。
   - 根据市场趋势（均线）和RSI指标动态调整目标比例。

2. **ATR止损与止盈**：
   - 使用ATR计算动态止损和止盈价格，适应市场波动。
   - 止损和止盈触发时自动下市价单平仓，确保及时止损或锁定利润。

3. **回测模式**：
   - 支持从CSV文件加载历史K线数据进行回测。
   - 模拟交易费用（默认0.1%）和滑点，优化关键参数（如目标比例、止损/止盈倍数）。

4. **风险管理**：
   - 动态仓位调整，限制单笔交易风险（最大仓位比例3%）。
   - 紧急停止机制，应对极端市场情况。
   - 冷却期（默认60秒）避免高频交易。

5. **异步架构**：
   - 使用`asyncio`和`aiohttp`确保高效的市场数据更新和订单执行。

---

## 安装与依赖

### 环境要求
- Python 3.8+
- 依赖库：
  ```bash
  pip install aiohttp numpy pandas
  ```

### 安装步骤
1. 下载或克隆本代码文件（`secure_trading_bot.py`）。
2. 安装依赖库：
   ```bash
   pip install -r requirements.txt
   ```
   （如无`requirements.txt`，直接安装上述库）
3. 确保有有效的Binance API密钥（`api_key`和`api_secret`）。

---

## 配置说明

配置文件位于`main`函数中的`config`字典，以下为关键参数说明：

| 参数名 | 描述 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `symbol` | 交易对 | `BTCUSDT` | 目前仅支持BTC/USDT |
| `api_key` | Binance API密钥 | `YOUR_BINANCE_API_KEY` | 替换为您的API密钥 |
| `api_secret` | Binance API密钥 | `YOUR_BINANCE_SECRET` | 替换为您的API密钥 |
| `testnet` | 是否使用测试网 | `True` | 建议先在测试网运行 |
| `target_ratio` | 目标BTC比例 | `0.5` | 投资组合中BTC价值的比例（0-1） |
| `balance_threshold` | 再平衡阈值 | `0.03` | 触发再平衡的比例偏差 |
| `min_trade_amount` | 最小交易量 | `0.001` | 最小BTC交易量 |
| `max_position_ratio` | 最大仓位比例 | `0.03` | 单笔交易最大占总资产的比例 |
| `stop_loss_atr_multiplier` | 止损ATR倍数 | `2.0` | 止损距离 = ATR × 倍数 |
| `take_profit_atr_multiplier` | 止盈ATR倍数 | `3.0` | 止盈距离 = ATR × 倍数 |
| `atr_period` | ATR计算周期 | `14` | ATR计算的K线数量 |
| `backtest_mode` | 是否启用回测 | `False` | 设为`True`以运行回测 |
| `backtest_file` | 回测数据文件 | `btcusdt_1m.csv` | CSV文件路径（OHLCV格式） |
| `trading_fee_rate` | 交易费率 | `0.001` | 默认0.1% |
| `max_slippage_percent` | 最大滑点百分比 | `0.3` | 回测中滑点控制 |
| `cooldown_period` | 交易冷却时间 | `60` | 秒，防止高频交易 |

### CSV文件格式（回测模式）
回测需要提供OHLCV格式的CSV文件，包含以下列：
- `timestamp`：Unix时间戳（毫秒或秒）
- `open`：开盘价
- `high`：最高价
- `low`：最低价
- `close`：收盘价
- `volume`：成交量

示例：
```
timestamp,open,high,low,close,volume
1625097600000,35000.0,35100.0,34900.0,35050.0,100.0
1625097660000,35050.0,35150.0,35000.0,35100.0,120.0
...
```

---

## 使用方法

### 1. 配置API密钥
在`main`函数中，将`api_key`和`api_secret`替换为您的Binance API密钥：
```python
config = {
    'api_key': '您的API密钥',
    'api_secret': '您的API密钥',
    ...
}
```

### 2. 运行回测模式
1. 准备OHLCV格式的CSV文件（如`btcusdt_1m.csv`），确保包含至少3-6个月的1分钟K线数据。
2. 设置`backtest_mode: True`和`backtest_file`路径：
   ```python
   config = {
       'backtest_mode': True,
       'backtest_file': 'path/to/btcusdt_1m.csv',
       ...
   }
   ```
3. 运行脚本：
   ```bash
   python secure_trading_bot.py
   ```
4. 脚本将自动进行参数优化（`target_ratio`、`balance_threshold`、`stop_loss_atr_multiplier`、`take_profit_atr_multiplier`），输出最佳参数和回测结果。

### 3. 运行模拟模式
1. 设置`simulation_mode: True`和`testnet: True`：
   ```python
   config = {
       'simulation_mode': True,
       'testnet': True,
       ...
   }
   ```
2. 运行脚本：
   ```bash
   python secure_trading_bot.py
   ```
3. 观察模拟交易的表现，验证策略在测试网上的执行效果。

### 4. 运行实盘模式
1. 设置`simulation_mode: False`和`testnet: False`：
   ```python
   config = {
       'simulation_mode': False,
       'testnet': False,
       ...
   }
   ```
2. 使用优化后的参数（从回测结果中获取）。
3. 以小额资金（如$100-$500）开始，运行脚本：
   ```bash
   python secure_trading_bot.py
   ```
4. 实时监控日志（`secure_trading_bot.log`）和控制台输出。

### 5. 停止机器人
- 按`Ctrl+C`终止程序，机器人将执行优雅关闭，取消所有活动订单。
- 或设置`emergency_stop: True`触发紧急停止。

---

## 运行输出

运行时，机器人每20次循环输出状态报告，包括：
- 当前价格、投资组合比例、总价值
- 波动率、RSI、趋势强度
- 订单统计（总数、成功率、活跃订单、每日交易量）
- 止损和止盈触发次数
- 总盈利、平均滑点
- 当前余额（BTC和USDT）

示例日志：
```
2025-08-20 21:05:00,123 - INFO - === Enhanced Status Report ===
2025-08-20 21:05:00,124 - INFO - Price: 35050.00 | Ratio: 0.502 | Value: 13500.00
2025-08-20 21:05:00,125 - INFO - Volatility: 0.0150 | RSI: 55.2 | Trend: 0.123
2025-08-20 21:05:00,126 - INFO - Orders: 10 total, 80.0% success
2025-08-20 21:05:00,127 - INFO - Active: 1 orders, Daily: 5
2025-08-20 21:05:00,128 - INFO - Profit: 25.4321 USDT, Avg Slippage: 0.0023
2025-08-20 21:05:00,129 - INFO - Stop-loss triggers: 2, Take-profit triggers: 3
2025-08-20 21:05:00,130 - INFO - Balances - BTC: 0.200000, USDT: 6500.00
```

---

## 注意事项

1. **风险提示**：
   - 加密货币交易存在高风险，可能导致资金全损。
   - 实盘前必须在模拟模式和测试网上充分测试。
   - 初始资金建议小额（如$100-$500），并设置实时监控。

2. **回测建议**：
   - 使用至少3-6个月的1分钟K线数据，覆盖牛市、熊市和震荡市场。
   - 优化参数后，验证结果在不同时间段的稳定性。

3. **实盘部署**：
   - 确保API密钥安全，限制权限（仅启用交易权限）。
   - 设置紧急停止机制（手动或通过`emergency_stop`）。
   - 定期检查日志，监控止损/止盈触发和账户余额。

4. **常见问题**：
   - **回测无结果**：检查CSV文件格式和路径，确保包含所有必要列。
   - **订单未执行**：检查余额是否充足，确认`min_trade_amount`和`min_notional`符合交易所要求。
   - **止损/止盈未触发**：确保ATR周期和倍数设置合理，检查市场波动是否过低。

---

## 示例运行流程

1. **准备回测数据**：
   - 下载BTC/USDT 1分钟K线数据，保存为`btcusdt_1m.csv`。
2. **运行回测**：
   ```bash
   python secure_trading_bot.py
   ```
   - 查看回测结果，记录最佳参数（如`target_ratio=0.5`, `balance_threshold=0.03`, `stop_loss_atr_multiplier=2.0`, `take_profit_atr_multiplier=3.0`）。
3. **运行模拟模式**：
   - 更新`config`中的最佳参数，设置`simulation_mode: True`。
   - 运行脚本，观察1-2周的模拟交易表现。
4. **实盘部署**：
   - 更新`config`，设置`simulation_mode: False`, `testnet: False`。
   - 以小额资金运行，实时监控日志和账户状态。

---

## 维护与优化

1. **参数优化**：
   - 定期运行`run_backtest`以更新参数，适应市场变化。
   - 调整`param_grid`以测试更多参数组合。

2. **扩展功能**：
   - 添加多时间框架分析（如1小时、4小时均线）以提高信号准确性。
   - 引入市场深度检查，优化订单执行。

3. **监控与报警**：
   - 集成外部报警系统（如Telegram或邮件）以实时通知止损/止盈触发。

---

## 免责声明

本机器人仅为自动化交易工具，作者不对任何交易损失负责。用户需自行承担所有交易风险，并确保遵守当地法律法规。在使用前，请充分理解加密货币市场的风险并进行充分测试。