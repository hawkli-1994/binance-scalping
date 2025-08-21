#!/usr/bin/env python3
"""
Data preprocessing script for Yahoo Finance CSV files
Converts Yahoo Finance format to the format required by the trading bot
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def preprocess_yahoo_data(input_file, output_file, symbol):
    """Preprocess Yahoo Finance CSV data to required format"""
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        # Skip the first 3 header rows
        df = df.iloc[3:].reset_index(drop=True)
        
        # Set column names
        df.columns = ['timestamp', 'close', 'high', 'low', 'open', 'volume']
        
        # Convert datetime to timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing data
        df = df.dropna()
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        print(f"✓ Processed {input_file} -> {output_file}")
        print(f"  - Rows: {len(df)}")
        print(f"  - Date range: {datetime.fromtimestamp(df['timestamp'].min())} to {datetime.fromtimestamp(df['timestamp'].max())}")
        print(f"  - Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_file}: {e}")
        return False

def main():
    """Process all Yahoo Finance data files"""
    data_dir = "../data"
    processed_dir = "processed_data"
    
    # Create processed data directory
    os.makedirs(processed_dir, exist_ok=True)
    
    # File mappings
    file_mappings = [
        ("yahoo_eth_usd_15m.csv", "eth_15m.csv", "ETH-USD"),
        ("yahoo_btc_4h.csv", "btc_4h.csv", "BTC-USD"),
        ("yahoo_eth_4h.csv", "eth_4h.csv", "ETH-USD"),
        ("yahoo_doge_4h.csv", "doge_4h.csv", "DOGE-USD")
    ]
    
    print("🔄 Preprocessing Yahoo Finance data...")
    print("=" * 50)
    
    success_count = 0
    
    for input_file, output_file, symbol in file_mappings:
        input_path = os.path.join(data_dir, input_file)
        output_path = os.path.join(processed_dir, output_file)
        
        if os.path.exists(input_path):
            if preprocess_yahoo_data(input_path, output_path, symbol):
                success_count += 1
        else:
            print(f"⚠️  Input file not found: {input_path}")
    
    print("=" * 50)
    print(f"✅ Successfully processed {success_count}/{len(file_mappings)} files")
    
    if success_count > 0:
        print(f"\n📁 Processed data saved to: {processed_dir}/")
        print("\n🔧 You can now run backtests with:")
        for _, output_file, symbol in file_mappings:
            print(f"   python bot.py backtest --data-file processed_data/{output_file} --symbol {symbol.replace('-', '')}")

if __name__ == '__main__':
    main()