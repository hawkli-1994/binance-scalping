"""
Main entry point for the trading bot
"""

import asyncio
import logging
import sys
import os
from typing import Optional

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli import cli

def main():
    """Main entry point"""
    try:
        cli()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()