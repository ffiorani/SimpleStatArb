# Statistical Arbitrage Strategy

This repository contains an implementation of a basic trend-following Statistical Arbitrage Strategy. The core of the strategy is encapsulated in the `StatisticalArbitrage` class found in `SimpleStatArb.py`, which includes custom online functions for calculating running window statistics in real-time. This efficient approach is a step towards deploying the strategy in a live trading environment.

## Contents

- **`SimpleStatArb.py`**: Contains the `StatisticalArbitrage` class and online statistical functions.
- **`SimpleStatisticalArbitrage.ipynb`**: A Jupyter notebook demonstrating the strategy using historical exchange rate data.

## Strategy Overview

In `SimpleStatisticalArbitrage.ipynb`, we select several exchange rates as leading indicators to determine market trends. Following the methodology outlined in [*Learn Algorithmic Trading* by Ghosh and Donadio](https://www.amazon.co.uk/Learn-Algorithmic-Trading-algorithmic-strategies/dp/178934834X), we use the "CADUSD" pair as our primary trading instrument. The process includes:

1. **Correlation Tracking**: Monitoring the correlation between various exchange rates.
2. **Valuation Difference Time Series**: Analyzing the valuation differences over time.
3. **Weighted Average Calculation**: Computing a weighted average of leading exchange rates, weighted by their correlation with the "CADUSD" trading instrument.
4. **Portfolio Adjustment**: Adjusting the portfolio based on deviations of the trading instrument's price from the weighted average.

## Risk Management

The strategy incorporates robust risk management features based on market volatility and historical performance data, including:

- Maximum number of trades per time frame.
- Maximum allowable PnL loss within a time frame.
- Maximum position limits.

## Performance

The strategy's performance is showcased in `SimpleStatArb.ipynb`, where it achieved a Sharpe ratio of 1.26 over a three-year period.

## Getting Started

To explore and utilize the strategy, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/statistical-arbitrage.git
    ```
2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
3. Run the Jupyter notebook:
    ```sh
    jupyter notebook SimpleStatisticalArbitrage.ipynb
    ```

## Contributing

We welcome contributions to enhance the strategy and its implementation. Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## References

- Ghosh, S., & Donadio, S. *Learn Algorithmic Trading*. Packt Publishing. [Amazon Link](https://www.amazon.co.uk/Learn-Algorithmic-Trading-algorithmic-strategies/dp/178934834X)

---
