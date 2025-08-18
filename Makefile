test_btc_4h:
	python backtest.py --data-file data/yahoo_btc_4h.csv --symbol ETHUSD --quantity 0.01 --take-profit 1 --direction BUY --max-orders 75

test_doge_4h:
	python backtest.py --data-file data/yahoo_doge_4h.csv --symbol DOGEUSD --quantity 500 --take-profit 0.0001 --direction BUY --max-orders 75

test_eth_4h:
	python backtest.py --data-file data/yahoo_eth_4h.csv --symbol ETHUSD --quantity 0.01 --take-profit 1 --direction BUY --max-orders 75