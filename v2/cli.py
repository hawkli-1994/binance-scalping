"""
Command Line Interface

This module provides a unified CLI for the trading bot with multiple subcommands.
"""

import asyncio
import logging
import sys
import os
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich.align import Align

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ConfigManager, config_manager, get_config, get_trading_engine_config
from trading_engine import TradingEngine, TradingMode
from strategies import StrategyManager, MovingAverageStrategy, MeanReversionStrategy, GridTradingStrategy, StrategyConfig

console = Console()

def setup_logging(config):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.FileHandler(config.logging.file_path),
            logging.StreamHandler() if config.logging.console_output else logging.NullHandler()
        ]
    )

@click.group()
@click.option('--config', '-c', default='config.json', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Binance Trading Bot - Advanced cryptocurrency trading system"""
    ctx.ensure_object(dict)
    ctx.obj['config_file'] = config
    ctx.obj['verbose'] = verbose
    
    # Initialize config manager
    global config_manager
    config_manager = ConfigManager(config)
    
    if verbose:
        config_manager.config.logging.level = 'DEBUG'
    
    setup_logging(config_manager.config)

@cli.command()
@click.option('--symbol', '-s', default='BTCUSDT', help='Trading symbol')
@click.option('--mode', '-m', type=click.Choice(['live', 'simulation', 'backtest']), 
              default='simulation', help='Trading mode')
@click.option('--data-file', '-d', help='Data file for backtesting')
@click.option('--strategies', '-S', multiple=True, help='Strategies to enable')
@click.pass_context
def run(ctx, symbol, mode, data_file, strategies):
    """Run the trading bot"""
    try:
        # Update configuration
        updates = {
            'trading': {'symbol': symbol, 'mode': mode},
            'backtest': {'data_file': data_file or ''}
        }
        
        if strategies:
            for strategy_name in strategies:
                strategy_config = config_manager.get_strategy_config(strategy_name)
                if strategy_config:
                    strategy_config.enabled = True
        
        config_manager.update_config(updates)
        config = config_manager.get_config()
        
        console.print(Panel.fit(
            f"[bold green]Starting Trading Bot[/bold green]\n"
            f"Symbol: {symbol}\n"
            f"Mode: {mode}\n"
            f"Strategies: {', '.join(strategies) if strategies else 'All enabled'}",
            title="Bot Configuration"
        ))
        
        # Initialize trading engine
        engine_config = get_trading_engine_config()
        engine = TradingEngine(engine_config)
        
        # Initialize strategy manager
        strategy_manager = StrategyManager(engine)
        
        # Add strategies
        for strategy_config in config.strategies:
            if not strategies or strategy_config.name in strategies:
                if strategy_config.enabled:
                    if strategy_config.type == 'moving_average':
                        params = strategy_config.parameters
                        strategy = MovingAverageStrategy(
                            strategy_config, engine,
                            fast_period=params.get('fast_period', 10),
                            slow_period=params.get('slow_period', 30)
                        )
                    elif strategy_config.type == 'mean_reversion':
                        params = strategy_config.parameters
                        strategy = MeanReversionStrategy(
                            strategy_config, engine,
                            period=params.get('period', 20),
                            std_dev_threshold=params.get('std_dev_threshold', 2.0)
                        )
                    elif strategy_config.type == 'grid':
                        params = strategy_config.parameters
                        strategy = GridTradingStrategy(
                            strategy_config, engine,
                            grid_spacing=params.get('grid_spacing', 0.01),
                            grid_levels=params.get('grid_levels', 10)
                        )
                    else:
                        continue
                    
                    strategy_manager.add_strategy(strategy)
        
        # Run the bot
        asyncio.run(_run_bot(engine, strategy_manager, mode))
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

async def _run_bot(engine, strategy_manager, mode):
    """Run the trading bot"""
    try:
        # Initialize engine
        if not await engine.initialize():
            console.print("[bold red]Failed to initialize trading engine[/bold red]")
            return
        
        # Start strategies
        await strategy_manager.start_all_strategies()
        
        # Start engine
        if mode == 'backtest':
            await _run_backtest(engine, strategy_manager)
        else:
            await _run_live(engine, strategy_manager)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Bot stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
    finally:
        await engine.stop()
        await strategy_manager.stop_all_strategies()

async def _run_live(engine, strategy_manager):
    """Run live trading with dashboard"""
    console.print("[green]Starting live trading...[/green]")
    
    # Create dashboard layout
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="stats"),
        Layout(name="strategies"),
        Layout(name="orders")
    )
    
    with Live(layout, refresh_per_second=1) as live:
        while engine.is_running:
            try:
                # Update header
                layout["header"].update(Panel.fit(
                    f"[bold blue]Trading Bot Live[/bold blue] | "
                    f"Mode: {engine.mode.value} | "
                    f"Symbol: {engine.config.get('symbol', 'BTCUSDT')}"
                ))
                
                # Update stats
                account_info = await engine.get_account_info()
                stats_table = Table(title="Account Stats")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                
                if account_info:
                    stats_table.add_row("Total Value", f"${account_info.get('total_value', 0):.2f}")
                    stats_table.add_row("Current Price", f"${account_info.get('ticker', {}).get('last', 0):.2f}")
                    
                    balances = account_info.get('balances', {})
                    for asset, balance in balances.items():
                        stats_table.add_row(f"{asset} Balance", f"{balance.free:.6f}")
                
                layout["stats"].update(stats_table)
                
                # Update strategies
                strategies_table = Table(title="Strategies")
                strategies_table.add_column("Name", style="cyan")
                strategies_table.add_column("Status", style="green")
                strategies_table.add_column("Position", style="yellow")
                strategies_table.add_column("Trades", style="magenta")
                
                for name, strategy in strategy_manager.strategies.items():
                    status = "🟢 Active" if strategy.is_active else "🔴 Inactive"
                    strategies_table.add_row(
                        name,
                        status,
                        f"{strategy.current_position:.6f}",
                        str(len(strategy.trade_history))
                    )
                
                layout["strategies"].update(strategies_table)
                
                # Update orders
                try:
                    open_orders = await engine.client.get_open_orders()
                    orders_table = Table(title="Open Orders")
                    orders_table.add_column("ID", style="cyan")
                    orders_table.add_column("Side", style="green")
                    orders_table.add_column("Quantity", style="yellow")
                    orders_table.add_column("Price", style="magenta")
                    
                    for order in open_orders[:10]:  # Show first 10 orders
                        orders_table.add_row(
                            order.id[:10] + "...",
                            order.side.value,
                            f"{order.quantity:.6f}",
                            f"{order.price:.2f}" if order.price else "Market"
                        )
                    
                    layout["orders"].update(orders_table)
                except:
                    layout["orders"].update(Panel("No open orders"))
                
                # Update footer
                layout["footer"].update(Panel.fit(
                    "Press Ctrl+C to stop | " +
                    f"Active Strategies: {len(strategy_manager.active_strategies)} | " +
                    f"Total Orders: {len(engine.order_history)}"
                ))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                console.print(f"[red]Dashboard error: {e}[/red]")
                await asyncio.sleep(5)

async def _run_backtest(engine, strategy_manager):
    """Run backtest with progress"""
    console.print("[green]Starting backtest...[/green]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running backtest...", total=None)
        
        # Start engine
        await engine.start()
        
        # Update progress
        while engine.is_running:
            if isinstance(engine.client, engine.client.__class__):
                progress.update(task, description=f"Processing... {engine.client.current_index}/{len(engine.client.data)}")
            await asyncio.sleep(0.1)
    
    # Show results
    results = await engine.get_backtest_results()
    if results:
        _display_backtest_results(results)

def _display_backtest_results(results):
    """Display backtest results"""
    console.print("\n[bold blue]Backtest Results[/bold blue]")
    
    results_table = Table()
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")
    
    results_table.add_row("Total Trades", str(results.get('total_trades', 0)))
    results_table.add_row("Total Profit", f"${results.get('total_profit', 0):.2f}")
    results_table.add_row("Total Fees", f"${results.get('total_fees', 0):.2f}")
    
    console.print(results_table)

@cli.command()
@click.option('--data-file', '-d', required=True, help='CSV data file for backtesting')
@click.option('--symbol', '-s', default='BTCUSDT', help='Trading symbol')
@click.option('--initial-balance', '-b', type=float, default=10000, help='Initial balance')
@click.option('--strategies', '-S', multiple=True, help='Strategies to test')
@click.option('--output', '-o', help='Output file for results')
@click.pass_context
def backtest(ctx, data_file, symbol, initial_balance, strategies, output):
    """Run backtest with specified parameters"""
    try:
        console.print(f"[blue]Running backtest with {data_file}[/blue]")
        
        # Update configuration
        updates = {
            'trading': {'symbol': symbol, 'mode': 'backtest'},
            'backtest': {'data_file': data_file, 'initial_balance': initial_balance}
        }
        
        config_manager.update_config(updates)
        
        # Run backtest
        engine_config = get_trading_engine_config()
        engine_config['mode'] = 'backtest'  # Force backtest mode
        engine = TradingEngine(engine_config)
        
        strategy_manager = StrategyManager(engine)
        
        # Add strategies
        config = config_manager.get_config()
        for strategy_config in config.strategies:
            if not strategies or strategy_config.name in strategies:
                if strategy_config.type == 'moving_average':
                    params = strategy_config.parameters
                    strategy = MovingAverageStrategy(
                        strategy_config, engine,
                        fast_period=params.get('fast_period', 10),
                        slow_period=params.get('slow_period', 30)
                    )
                elif strategy_config.type == 'mean_reversion':
                    params = strategy_config.parameters
                    strategy = MeanReversionStrategy(
                        strategy_config, engine,
                        period=params.get('period', 20),
                        std_dev_threshold=params.get('std_dev_threshold', 2.0)
                    )
                else:
                    continue
                
                strategy_manager.add_strategy(strategy)
        
        # Run backtest
        asyncio.run(_run_backtest(engine, strategy_manager))
        
        # Save results if output file specified
        if output:
            results = engine.get_backtest_results()
            import json
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)

@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration"""
    try:
        config = config_manager.get_config()
        
        # Create configuration table
        config_table = Table(title="Current Configuration")
        config_table.add_column("Section", style="cyan")
        config_table.add_column("Parameter", style="yellow")
        config_table.add_column("Value", style="green")
        
        # Exchange config
        config_table.add_row("Exchange", "Exchange ID", config.exchange.exchange_id)
        config_table.add_row("Exchange", "Testnet", str(config.exchange.testnet))
        config_table.add_row("Exchange", "API Key", config.exchange.api_key[:20] + "..." if config.exchange.api_key else "Not set")
        
        # Trading config
        config_table.add_row("Trading", "Symbol", config.trading.symbol)
        config_table.add_row("Trading", "Mode", config.trading.mode)
        config_table.add_row("Trading", "Max Position Size", str(config.trading.max_position_size))
        
        # Strategies
        for strategy in config.strategies:
            config_table.add_row("Strategies", strategy.name, f"{strategy.type} ({'Enabled' if strategy.enabled else 'Disabled'})")
        
        console.print(config_table)
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

@cli.command()
@click.option('--strategy', '-s', required=True, help='Strategy name')
@click.option('--enable/--disable', default=True, help='Enable or disable strategy')
@click.pass_context
def strategy(ctx, strategy, enable):
    """Enable or disable a strategy"""
    try:
        if enable:
            if config_manager.enable_strategy(strategy):
                console.print(f"[green]Strategy '{strategy}' enabled[/green]")
            else:
                console.print(f"[red]Strategy '{strategy}' not found[/red]")
        else:
            if config_manager.disable_strategy(strategy):
                console.print(f"[green]Strategy '{strategy}' disabled[/green]")
            else:
                console.print(f"[red]Strategy '{strategy}' not found[/red]")
                
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

@cli.command()
@click.pass_context
def status(ctx):
    """Show trading bot status"""
    try:
        config = config_manager.get_config()
        
        # Create status panel
        status_text = Text()
        status_text.append("Trading Bot Status\n", style="bold blue")
        status_text.append(f"Configuration: {ctx.obj['config_file']}\n", style="cyan")
        status_text.append(f"Trading Mode: {config.trading.mode}\n", style="cyan")
        status_text.append(f"Symbol: {config.trading.symbol}\n", style="cyan")
        status_text.append(f"Exchange: {config.exchange.exchange_id}\n", style="cyan")
        status_text.append(f"Testnet: {config.exchange.testnet}\n", style="cyan")
        
        # Count enabled strategies
        enabled_strategies = [s.name for s in config.strategies if s.enabled]
        status_text.append(f"Enabled Strategies: {', '.join(enabled_strategies)}\n", style="green")
        
        console.print(Panel(status_text, title="Bot Status"))
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

@cli.command()
@click.pass_context
def init(ctx):
    """Initialize configuration file"""
    try:
        config_file = ctx.obj['config_file']
        
        if os.path.exists(config_file):
            if not click.confirm(f"Configuration file {config_file} already exists. Overwrite?"):
                return
        
        config_manager.create_sample_config(config_file)
        console.print(f"[green]Configuration file created: {config_file}[/green]")
        console.print("[yellow]Please edit the configuration file with your API keys and preferences.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")

@cli.command()
@click.option('--file', '-f', default='config.json', help='Configuration file to validate')
@click.pass_context
def validate(ctx, file):
    """Validate configuration file"""
    try:
        # Test loading configuration
        test_manager = ConfigManager(file)
        config = test_manager.get_config()
        
        console.print(f"[green]Configuration file {file} is valid![/green]")
        
        # Show summary
        console.print(f"Exchange: {config.exchange.exchange_id}")
        console.print(f"Symbol: {config.trading.symbol}")
        console.print(f"Mode: {config.trading.mode}")
        console.print(f"Strategies: {len([s for s in config.strategies if s.enabled])} enabled")
        
    except Exception as e:
        console.print(f"[bold red]Configuration validation failed: {e}[/bold red]")
        sys.exit(1)

if __name__ == '__main__':
    cli()