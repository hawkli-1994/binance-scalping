const Binance = require('node-binance-api');
const uuid = require('uuid');

// Configuration
const config = {
  symbol: 'BTCUSDT',
  apiKey: 'YOUR_API_KEY', // Replace with your Binance API key
  apiSecret: 'YOUR_API_SECRET', // Replace with your Binance API secret
  targetRatio: 0.5, // Target BTC/(BTC+USDT) ratio
  balanceThreshold: 0.02, // Rebalance if ratio deviates by ±2%
  burstThreshold: 2, // Price movement threshold (in standard deviations)
  minTradeAmount: 0.001, // Minimum trade size (BTC)
  maxPosition: 0.1, // Maximum position size (BTC)
  pollInterval: 5000, // Polling interval (ms)
  stopLoss: 0.05, // Stop-loss threshold (5% below entry price)
  volumeLookback: 10, // Number of trades for volume calculation
  priceLookback: 20, // Number of prices for burst detection
};

// Initialize Binance client
const binance = new Binance().options({
  APIKEY: config.apiKey,
  APISECRET: config.apiSecret,
  useServerTime: true,
});

// Trading strategy class
class TradingBot {
  constructor() {
    this.numTick = 0;
    this.lastTradeId = 0;
    this.volume = 0;
    this.orderBook = { asks: [], bids: [] };
    this.prices = [];
    this.account = null;
    this.btc = 0;
    this.usdt = 0;
    this.p = 0.5;
    this.activeOrderId = null;
    this.lastPrice = 0;
    this.entryPrice = 0;
  }

  // Calculate simple moving average
  calculateSMA(data) {
    return data.reduce((sum, val) => sum + val, 0) / data.length;
  }

  // Calculate standard deviation
  calculateSD(data, mean) {
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    return Math.sqrt(variance);
  }

  // Update trade volume
  async updateTrades() {
    try {
      const trades = await binance.trades(config.symbol, { limit: config.volumeLookback });
      let totalVolume = 0;
      for (const trade of trades) {
        const tradeId = parseInt(trade.id);
        if (tradeId > this.lastTradeId) {
          this.lastTradeId = tradeId;
          totalVolume += parseFloat(trade.qty);
        }
      }
      this.volume = 0.7 * this.volume + 0.3 * totalVolume;
      if (this.prices.length === 0) {
        this.prices = Array(config.priceLookback).fill(parseFloat(trades[0].price));
      }
    } catch (error) {
      console.error('Error fetching trades:', error.message);
    }
  }

  // Update order book and prices
  async updateOrderBook() {
    try {
      const depth = await binance.depth(config.symbol, { limit: 5 });
      this.orderBook = {
        bids: depth.bids.map(([price, qty]) => ({ price: parseFloat(price), qty: parseFloat(qty) })),
        asks: depth.asks.map(([price, qty]) => ({ price: parseFloat(price), qty: parseFloat(qty) })),
      };
      if (this.orderBook.bids.length < 3 || this.orderBook.asks.length < 3) {
        return;
      }
      const midPrice = (this.orderBook.bids[0].price + this.orderBook.asks[0].price) / 2;
      this.prices.shift();
      this.prices.push(midPrice);
      this.lastPrice = midPrice;
    } catch (error) {
      console.error('Error fetching order book:', error.message);
    }
  }

  // Update account balance and portfolio ratio
  async balanceAccount() {
    try {
      const accountInfo = await binance.account();
      this.account = accountInfo;
      this.btc = parseFloat(accountInfo.balances.find(b => b.asset === 'BTC').free);
      this.usdt = parseFloat(accountInfo.balances.find(b => b.asset === 'USDT').free);
      const btcValue = this.btc * this.lastPrice;
      this.p = btcValue / (btcValue + this.usdt);

      if (this.p < config.targetRatio - config.balanceThreshold) {
        console.log(`Rebalancing: Portfolio ratio ${this.p.toFixed(3)} too low`);
        const buyAmount = Math.min((this.usdt * 0.01) / this.lastPrice, config.maxPosition);
        if (buyAmount >= config.minTradeAmount) {
          await this.placeOrder('BUY', buyAmount, this.orderBook.bids[0].price);
        }
      } else if (this.p > config.targetRatio + config.balanceThreshold) {
        console.log(`Rebalancing: Portfolio ratio ${this.p.toFixed(3)} too high`);
        const sellAmount = Math.min(this.btc * 0.01, config.maxPosition);
        if (sellAmount >= config.minTradeAmount) {
          await this.placeOrder('SELL', sellAmount, this.orderBook.asks[0].price);
        }
      }

      // Cancel old orders
      const openOrders = await binance.openOrders(config.symbol);
      for (const order of openOrders) {
        if (order.orderId !== this.activeOrderId) {
          await binance.cancel(config.symbol, order.orderId);
        }
      }
    } catch (error) {
      console.error('Error fetching account:', error.message);
    }
  }

  // Place an order
  async placeOrder(side, quantity, price) {
    try {
      const order = await binance.order({
        symbol: config.symbol,
        side,
        type: 'LIMIT',
        quantity: quantity.toFixed(8),
        price: price.toFixed(2),
      });
      this.activeOrderId = order.orderId;
      this.entryPrice = side === 'BUY' ? price : this.entryPrice;
      console.log(`Placed ${side} order: ${quantity.toFixed(8)} @ ${price.toFixed(2)}`);
      return order;
    } catch (error) {
      console.error(`Error placing ${side} order:`, error.message);
    }
  }

  // Check stop-loss
  async checkStopLoss() {
    if (this.entryPrice && this.lastPrice && this.btc > 0) {
      if (this.lastPrice < this.entryPrice * (1 - config.stopLoss)) {
        console.log('Stop-loss triggered');
        await this.placeOrder('SELL', Math.min(this.btc, config.maxPosition), this.orderBook.asks[0].price);
        this.entryPrice = 0; // Reset entry price
      }
    }
  }

  // Main polling loop
  async poll() {
    this.numTick++;
    await this.updateTrades();
    await this.updateOrderBook();
    await this.balanceAccount();
    await this.checkStopLoss();

    if (this.prices.length < config.priceLookback) {
      return;
    }

    const sma = this.calculateSMA(this.prices);
    const sd = this.calculateSD(this.prices, sma);
    const burstPrice = sd * config.burstThreshold;
    let tradeAmount = 0;
    let side = null;

    if (this.lastPrice > sma + burstPrice) {
      side = 'BUY';
      tradeAmount = Math.min((this.usdt * 0.5) / this.lastPrice, config.maxPosition);
    } else if (this.lastPrice < sma - burstPrice) {
      side = 'SELL';
      tradeAmount = Math.min(this.btc * 0.5, config.maxPosition);
    }

    // Adjust trade amount based on volume
    tradeAmount *= Math.min(this.volume / 1000, 1); // Scale by volume (arbitrary 1000 as reference)
    if (this.numTick < 10) {
      tradeAmount *= 0.8; // Reduce size for initial ticks
    }

    if (side && tradeAmount >= config.minTradeAmount) {
      const price = side === 'BUY' ? this.orderBook.bids[0].price : this.orderBook.asks[0].price;
      const order = await this.placeOrder(side, tradeAmount, price);
      if (order) {
        let remaining = tradeAmount;
        while (remaining >= config.minTradeAmount) {
          const orderStatus = await binance.orderStatus(config.symbol, order.orderId);
          if (orderStatus.status === 'FILLED') {
            break;
          } else if (orderStatus.status === 'PARTIALLY_FILLED') {
            remaining -= parseFloat(orderStatus.executedQty);
          } else {
            await binance.cancel(config.symbol, order.orderId);
            break;
          }
          await new Promise(resolve => setTimeout(resolve, 200));
        }
        this.activeOrderId = null;
      }
    }

    console.log(`Tick: ${this.numTick}, Price: ${this.lastPrice.toFixed(2)}, Ratio: ${this.p.toFixed(3)}, Volume: ${this.volume.toFixed(2)}`);
  }

  // Start the bot
  async start() {
    console.log('Starting trading bot...');
    while (true) {
      await this.poll();
      await new Promise(resolve => setTimeout(resolve, config.pollInterval));
    }
  }
}

// Main function
async function main() {
  const bot = new TradingBot();
  await bot.start();
}

// Run the bot
main().catch(console.error);