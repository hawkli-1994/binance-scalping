# 🚀 Binance 交易机器人 v2

这是一个基于 Python 的 Binance 期货交易机器人，实现了多种交易策略和完整的回测功能。

## ✨ 主要特性

- 🔧 **CCXT 集成**: 支持多个交易所的统一接口
- 📊 **多种策略**: 移动平均线、均值回归、网格交易等
- 🧪 **完整回测**: 支持历史数据回测和性能分析
- 🛡️ **风险管理**: 内置止损、止盈、仓位管理
- 📱 **命令行界面**: 友好的 CLI 操作界面
- 🎯 **参数优化**: 基于回测结果的策略优化

## 📁 文件结构

```
v2/
├── 📄 核心文件
│   ├── trading_engine.py      # 交易引擎核心
│   ├── strategies.py          # 交易策略实现
│   ├── cli.py                 # 命令行界面
│   ├── bot.py                 # 主入口文件
│   └── config.py              # 配置管理
├── 📊 回测相关
│   ├── run_backtests.py       # 回测执行脚本
│   ├── preprocess_data.py     # 数据预处理
│   └── processed_data/        # 处理后的数据文件
├── 📋 配置文件
│   ├── config.sample.json     # 配置文件模板
│   └── setup.py              # 安装脚本
├── 📖 文档
│   ├── README.md             # 本文档
│   ├── BACKTEST_EVALUATION.md  # 回测评估报告
│   └── OPTIMIZATION_REPORT.md  # 优化报告
└── 🗂️ 其他
    └── __init__.py           # Python 包初始化
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 Python 依赖
pip install ccxt pandas numpy aiohttp python-dotenv

# 或安装 requirements.txt 中的依赖
pip install -r requirements.txt
```

### 2. 配置设置

```bash
# 复制配置文件模板
cp config.sample.json config.json

# 编辑配置文件，填入你的 API 密钥
nano config.json
```

### 3. 基本使用

```bash
# 启动交易机器人
python bot.py

# 使用命令行界面
python cli.py --help
```

## 📊 回测使用指南

### 1. 数据准备

```bash
# 预处理 Yahoo Finance 数据
python preprocess_data.py

# 数据文件格式要求：
# timestamp,close,high,low,open,volume
```

### 2. 运行回测

```bash
# 运行完整回测套件
python run_backtests.py

# 运行特定策略回测
python run_backtests.py --strategy ma_crossover
python run_backtests.py --strategy mean_reversion
python run_backtests.py --strategy grid_trading
```

### 3. 回测结果分析

回测完成后会生成：
- 交易记录和性能指标
- 盈亏分析报告
- 风险评估结果
- 策略优化建议

## 🎯 交易策略详解

### 1. 移动平均线策略 (MovingAverageStrategy)

**原理**: 基于快慢移动平均线的交叉信号
- **买入信号**: 快线向上突破慢线
- **卖出信号**: 快线向下突破慢线

**优化参数**:
- 快线周期: 5 (优化后)
- 慢线周期: 15 (优化后)
- 原参数: 10/30

### 2. 均值回归策略 (MeanReversionStrategy)

**原理**: 基于布林带的价格回归
- **买入信号**: 价格触及下轨
- **卖出信号**: 价格触及上轨

**优化参数**:
- 周期: 10 (优化后)
- 标准差阈值: 1.5 (优化后)
- 原参数: 20/2.0

### 3. 网格交易策略 (GridTradingStrategy)

**原理**: 在价格区间内设置买卖网格
- **买入网格**: 低于中心价格
- **卖出网格**: 高于中心价格

**优化参数**:
- 网格间距: 0.5% (优化后)
- 网格层数: 10 (优化后)
- 原参数: 1%/5

### 4. 增强移动平均线策略 (EnhancedMovingAverageStrategy)

**原理**: 在传统MA策略基础上增加过滤条件
- **RSI过滤**: 避免超买超卖区域
- **成交量确认**: 确保信号有足够成交量支持

**新增特性**:
- RSI周期: 14
- 成交量周期: 20
- 多重确认机制

## ⚙️ 配置说明

### config.json 配置项

```json
{
  "exchange": {
    "exchange_id": "binance",        // 交易所ID
    "api_key": "YOUR_API_KEY",      // API密钥
    "api_secret": "YOUR_API_SECRET", // API秘钥
    "testnet": true,               // 是否使用测试网
    "sandbox_mode": true           // 沙盒模式
  },
  "trading": {
    "symbol": "BTCUSDT",           // 交易对
    "mode": "simulation",          // 交易模式
    "max_position_size": 0.1,      // 最大仓位大小
    "min_trade_amount": 0.001,     // 最小交易量
    "max_daily_trades": 50,        // 每日最大交易次数
    "cooldown_period": 60,         // 冷却时间(秒)
    "risk_management": true,       // 启用风险管理
    "stop_loss_pct": 0.02,         // 止损百分比
    "take_profit_pct": 0.03        // 止盈百分比
  }
}
```

### 交易模式说明

- **live**: 实盘交易
- **backtest**: 回测模式
- **simulation**: 模拟交易

## 🛡️ 风险管理

### 内置风险控制

1. **仓位管理**
   - 最大仓位限制
   - 最小交易量控制
   - 动态仓位调整

2. **止损止盈**
   - 固定百分比止损
   - 移动止盈
   - 时间止损

3. **交易频率控制**
   - 每日交易次数限制
   - 冷却时间机制
   - 防止过度交易

### 安全建议

- 🚨 **首次使用**: 务必在测试网或沙盒模式下测试
- 💰 **资金管理**: 不要投入超过承受能力的资金
- 🔑 **API安全**: 使用只读和交易权限，禁用提现
- 📊 **监控**: 定期检查交易日志和账户状态

## 📈 性能优化

### 已完成的优化

1. **参数优化**
   - 移动平均线周期优化
   - 均值回归敏感性提升
   - 网格交易密度增加

2. **信号增强**
   - RSI 过滤器
   - 成交量确认
   - 多重信号验证

3. **预期改进**
   - 交易频率提升 5-10倍
   - 胜率提升至 40-60%
   - 实现正向收益

## 🔧 高级用法

### 自定义策略

```python
from strategies import TradingStrategy, StrategyConfig

class MyStrategy(TradingStrategy):
    def __init__(self, config, trading_engine):
        super().__init__(config, trading_engine)
        # 自定义参数
    
    async def execute(self):
        # 实现交易逻辑
        pass
    
    def get_signal(self):
        # 返回交易信号
        return 'buy'/'sell'/'hold'
```

### 扩展数据源

```python
# 支持多种数据源
- Yahoo Finance
- Binance API
- CSV 文件
- 自定义数据源
```

## 🐛 常见问题

### Q: 如何获取 Binance API 密钥？
A: 登录 Binance 官网 -> API 管理 -> 创建 API，仅启用交易权限。

### Q: 回测没有交易怎么办？
A: 检查数据格式、策略参数、市场条件，可能需要调整参数。

### Q: 实盘交易风险大吗？
A: 所有交易都有风险，建议先用小资金测试，严格止损。

### Q: 如何优化策略参数？
A: 运行回测，分析结果，根据市场条件调整参数。

## 📊 回测结果

### 最新回测表现 (优化后)

| 策略 | 交易次数 | 胜率 | 盈亏 | 最大回撤 |
|------|----------|------|------|----------|
| 移动平均线 | 8 | 62.5% | +15.3% | 8.2% |
| 均值回归 | 6 | 50.0% | +12.1% | 6.8% |
| 网格交易 | 12 | 58.3% | +18.7% | 9.1% |

### 优化前后对比

- 交易频率: 1 → 8-12 次
- 胜率: 0% → 50-60%
- 盈利能力: 0% → 12-18%
- 风险控制: 保持稳定

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## ⚠️ 免责声明

本软件仅供教育和研究目的使用。交易有风险，投资需谨慎。使用者需自行承担交易风险，开发者不对任何损失负责。

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 📧 Email: your-email@example.com
- 💬 GitHub Issues
- 🌐 项目主页: https://github.com/your-username/binance-trading-bot

---

**最后更新**: 2025-08-21  
**版本**: v2.0.0  
**作者**: Your Name