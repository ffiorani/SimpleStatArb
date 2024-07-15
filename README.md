# SimpleStatArb: Implementation of a statistical arbitrage strategy in python.

In this repo I implement a basic trend-following Statistical Arbitrage Strategy. I implement a Statistical Arbitrage class in SimpleStatArb.py. I implement my own online functions to compute my running window statistics live. Not only this is efficient, but is a step towards deploying the strategy.

In SimpleStatisticalArbitrage.ipynb, I select a few exchange rates as the leading instruments that will determine the trend. Following "Learn Algorithmic Trading" by Ghosh and Donadio (https://www.amazon.co.uk/Learn-Algorithmic-Trading-algorithmic-strategies/dp/178934834X), I use "CADUSD" as trading instrument. I track the correlation between the different exchange rates and look at the valuation difference time series. I compute a weighted average of the leading exchange prices weighted by correlation with the trading instrument. If the price of the trading instrument deviates from this average, I adjust my portfolio accordingly.

I implemented strategy adjustment and risk management base on the volatility of the market and historical data of the strategy, such as maximum numbers of trade per time frame, maximum pnl loss within time frame and maximum position.
