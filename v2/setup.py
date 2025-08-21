"""
Setup script for Binance Trading Bot v2
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="binance-trading-bot",
    version="2.0.0",
    author="Trading Bot Team",
    author_email="support@tradingbot.com",
    description="Advanced cryptocurrency trading system with CCXT integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/binance-trading-bot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-bot=bot:main",
            "tbot=bot:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords="trading, binance, cryptocurrency, bot, ccxt, backtesting",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/binance-trading-bot/issues",
        "Source": "https://github.com/yourusername/binance-trading-bot",
        "Documentation": "https://binance-trading-bot.readthedocs.io/",
    },
)