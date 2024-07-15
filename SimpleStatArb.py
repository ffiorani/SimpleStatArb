from collections import deque
from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: need to restructure the program: possible reason why the correlations are not being computed correctly is that they should compute the correlation of the actual prices, not of the price deviations. See if changing that solves the problem(the smas are still for just the prices, right? Is that why we see correlations not between 0 and 1? or is it numerical instability?).

OUT_OF_RANGE_ERROR_MESSAGE = "curr_num_entries cannot be greater than n + 1"

def compute_rolling_mean_online(prev_mean, n, curr_num_entries, new_price, old_price = None):
    """
    @prev_mean: the mean at the previous time step
    @n: the period for the rolling mean
    @curr_num_entries: the current number of entries in the rolling mean after the new entry has been added. If less than n, the mean is computed as the average of the current number of entries
    @new_price: the new price
    @old_price: the price to be removed from the rolling mean
    
    Returns the updated rolling mean with window n given the new price and the old price to be removed"""
    
    if curr_num_entries < 2:
        return new_price
    
    if curr_num_entries > n:
        if curr_num_entries > n + 1:
            raise ValueError(OUT_OF_RANGE_ERROR_MESSAGE)
        if old_price is None:
            raise ValueError("old_price cannot be None when curr_num_entries > n")
        return prev_mean + (new_price - old_price) / n
    
    return (prev_mean * (curr_num_entries - 1) + new_price) / curr_num_entries


def compute_rolling_covariance_online(prev_covariance, prev_sma_x, prev_sma_y, n, curr_num_entries, x_new, y_new, x_old = None, y_old = None):
    """
    @prev_covariance: the covariance at the previous time step
    @prev_sma_x: the simple moving average of x at the previous time step
    @prev_sma_y: the simple moving average of y at the previous time step
    @n: the period for the rolling covariance
    @curr_num_entries: the current number of entries in the rolling covariance. If less than n, the covariance is computed as the average of the current number of entries
    @x_new: the new x value
    @y_new: the new y value
    @x_old: the x value to be removed from the rolling covariance
    @y_old: the y value to be removed from the rolling covariance
    
    Returns the updated rolling covariance with window n given the new x and y values and the old x and y values to be removed"""
    
    if curr_num_entries < 2:
        return 0

    if curr_num_entries <= n:
        addendum_factor = (curr_num_entries - 1) / curr_num_entries
        return ((curr_num_entries - 2) * prev_covariance + addendum_factor * (x_new * y_new + prev_sma_x * prev_sma_y - x_new * prev_sma_y - y_new * prev_sma_x)) / (curr_num_entries - 1)
    
    if curr_num_entries > n + 1:
        raise ValueError(OUT_OF_RANGE_ERROR_MESSAGE)
    if x_old is None or y_old is None:
        raise ValueError("x_old and y_old cannot be None when curr_num_entries > n")
    x_inner_average = prev_sma_x - x_old / n
    y_inner_average = prev_sma_y - y_old / n   
    x_diff = x_new - x_old
    y_diff = y_new - y_old
    return prev_covariance + (((n - 1) / n) * (x_new * y_new - x_old * y_old) - x_diff * y_inner_average - y_diff * x_inner_average) / (n - 1)


def compute_rolling_std_dev_online(prev_std_dev, sma, prev_sma, n, curr_num_entries, new_price, old_price = None):
    """
    @prev_std_dev: the standard deviation at the previous time step
    @sma: the simple moving average at the current time step
    @prev_sma: the simple moving average at the previous time step
    @n: the period for the rolling standard deviation
    @curr_num_entries: the current number of entries in the rolling standard deviation. If less than n, the standard deviation is computed as the average of the current number of entries
    @new_price: the new price
    @old_price: the price to be removed from the rolling standard deviation
    
    Returns the updated rolling standard deviation with window n given the new price and the old price to be removed"""
    
    if curr_num_entries < 2:
        return 0
    
    prev_variance = prev_std_dev ** 2
    if curr_num_entries > n:
        if curr_num_entries > n + 1:
            raise ValueError(OUT_OF_RANGE_ERROR_MESSAGE)
        if old_price is None:
            raise ValueError("old_price cannot be None when curr_num_entries > n")
        unbiasedness_factor = n - 1 #n - 1.5 + 1 / 8 / (n - 1)
        new_variance = prev_variance + (new_price - old_price) * (new_price - sma + old_price - prev_sma) / unbiasedness_factor
        
        return sqrt(new_variance)
    if curr_num_entries != 2:
        old_unbiasedness_factor = curr_num_entries - 2# curr_num_entries - 2.5 + 1 / 8 / (curr_num_entries - 2)
    else:
        old_unbiasedness_factor = 0

    new_unbiasedness_factor = curr_num_entries - 1 #curr_num_entries - 1.5 + 1 / 8 / (curr_num_entries - 1)
    
    addendum_factor = (curr_num_entries - 1) / curr_num_entries
    new_variance = (old_unbiasedness_factor * prev_variance + addendum_factor * (new_price ** 2 + prev_sma * (prev_sma - 2 * new_price))) / new_unbiasedness_factor
    
    assert new_variance >= 0
    return sqrt(new_variance)


def compute_max_drawdown(pnls):
    """
    @pnls: list containing the daily profits and losses
    
    Returns the maximum drawdown and the indices of the maximum pnl and the minimum pnl in the drawdown"""
    
    max_drawdown = 0
    max_pnl = 0
    max_pnl_index = 0
    drawdown_max_pnl_index = 0
    drawdown_min_pnl_index = 0
    for i in range(len(pnls)):
        pnl = pnls[i]
        if pnl > max_pnl:
            max_pnl = pnl
            max_pnl_index = i
        elif max_pnl - pnl > max_drawdown:
            max_drawdown = max_pnl - pnl
            drawdown_max_pnl_index = max_pnl_index
            drawdown_min_pnl_index = i

    return max_drawdown, drawdown_max_pnl_index, drawdown_min_pnl_index


def compute_rolling_sharpe_ratio(pnls, risk_free_rate = 0, window = 20):
    """
    @pnls: list containing the daily profits and losses
    @risk_free_rate: the risk free rate
    @window: the window for the rolling Sharpe ratio
    
    Returns a list containing the rolling Sharpe ratio given the profits and losses and the risk free rate. 
    The first window - 1 values are set to 0"""
    
    n = len(pnls)
    rolling_sharpe_ratio = [0 for _ in range(window - 1)]
    if n < 2:
        return 0
    for i in range(window, n + 1):
        mean_pnl = sum(pnls[i - window:i]) / window
        std_dev = sqrt(sum((pnl - mean_pnl) ** 2 for pnl in pnls[i - window:i]) / (window - 1))
        if std_dev != 0:
            rolling_sharpe_ratio.append((mean_pnl - risk_free_rate) / std_dev)
        else:
            rolling_sharpe_ratio.append(0)
    return rolling_sharpe_ratio


def compute_rolling_sortino_ratio(pnls, risk_free_rate = 0, window = 20):
    """
    @pnls: list containing the daily profits and losses
    @risk_free_rate: the risk free rate
    @window: the window for the rolling Sortino ratio
    
    Returns a list containing the rolling Sortino ratio given the profits and losses and the risk free rate.
    The first window - 1 values are set to 0"""
    
    n = len(pnls)
    if n < 2:
        return 0
    
    sortino_ratios = [0 for _ in range(window - 1)]
    for i in range(window, n + 1):
        mean_pnl = sum(pnls[i - window:i]) / window
        negative_pnls = [pnl for pnl in pnls[i - window:i] if pnl < 0]
        if len(negative_pnls) == 0:
            sortino_ratios.append(0)
            continue
        std_dev = sqrt(sum((pnl - mean_pnl) ** 2 for pnl in negative_pnls) / (window - 1))
        if std_dev != 0:
            sortino_ratios.append((mean_pnl - risk_free_rate) / std_dev)
        else:
            sortino_ratios.append(0)
    return sortino_ratios


def compute_sharpe_ratio(pnls, risk_free_rate = 0):
    """
    @pnls: list containing the daily profits and losses
    @risk_free_rate: the risk free rate, set to 0 by default
    
    Returns the Sharpe ratio given the profits and losses given the risk free rate"""
    
    if len(pnls) < 2:
        return 0
    
    mean_pnl = sum(pnls) / len(pnls)
    std_dev = sqrt(sum((pnl - mean_pnl) ** 2 for pnl in pnls) / (len(pnls) - 1))
    return (mean_pnl - risk_free_rate) / std_dev if std_dev != 0 else 0


def compute_sortino_ratio(pnls, risk_free_rate = 0):
    """
    @pnls: list containing the daily profits and losses
    @risk_free_rate: the risk free rate, set to 0 by default
    
    Returns the Sortino ratio given the profits and losses and the risk free rate"""
    
    if len(pnls) < 2:
        return 0
    
    negative_pnls = [pnl for pnl in pnls if pnl < 0]
    if len(negative_pnls) == 0:
        return 0
    
    mean_pnl = sum(pnls) / len(pnls)
    std_dev = sqrt(sum((pnl - mean_pnl) ** 2 for pnl in negative_pnls) / (len(negative_pnls) - 1))
    return (mean_pnl - risk_free_rate) / std_dev if std_dev != 0 else 0



class SimpleStatArb:
    """
    A simple trend-following statistical arbitrage strategy based on the correlation between the trading instrument and other symbols.
    The strategy is based on the following steps:
    1. Calculate the simple moving average of the prices of the trading instrument and the other symbols
    2. Compute how much the price deviates from the simple moving average
    3. Determine the correlation between the trading instrument and the other symbols
    4. Predict how much the trading instrument's price deviates from the simple moving average weighted the correlation with the other symbols
    5. Compute the difference between the predicted quantity just calculated and the actual price deviation from the simple moving average
    6. Use this signal to enter into a position
    
    Contains two modes of operation:
    1. Training mode, using the run_strategy_without_risk_optimization_based_on_historical_data method: the strategy is run with the input parameters, which are only updated using the standard deviation of the prices if the user sets the @std_dev_influence_on_strategy parameter. 
    2. Non-training mode using run_strategy_with_risk_optimization_based_on_historical_data: the strategy is run with the input parameters, using data up until the @training_end_date. The strategy is then run with the updated parameters based on the historical data, such as
    the maximum weekly and monthly losses, the maximum holding time, and the maximum total weekly volume. Such quantities are updated weekly based on the performance of the strategy. If one of these absolute thresholds risk parameters were to be exceeded, the strategy would be stopped. 
    
    If @std_dev_influence_on_strategy is set to a number different from zero, the standard deviation in the prices of the trading instrument updates the entry and the take profit threshold. The standard deviation is used to determine the influence of the standard deviation on the strategy.
    The higher the number, the more the standard deviation influences the strategy. The underlying assumption is that we want to be more cautious when the standard deviation is high. The mean of the rolling standard deviation is computed in the training mode and used as a reference point in the non-training mode.
    
    In training mode, the maximum weekly pnl, monthly pnl, total weekly volume, holding time, and position limit are determined based on the historical data. The strategy is then run in non-training mode, with the updated parameters based on the historical data. The strategy is stopped if one of the absolute thresholds is exceeded.
    
    The class also contains plotting functions to visualize the results of the strategy.
    
    Currently does not support an exit policy. The signal will either trigger buy or selling, modifying our position in the trading instrument. The class is designed to be used with coarse price data of the type that can be downloaded from Yahoo Finance.
    """
    
    def __init__(
            self, 
            prices, 
            trading_instrument_symbol, 
            sma_period = 50,
            std_dev_period = 50,
            price_dev_num_prices = 200, 
            value_for_entry = 0.01,
            value_to_take_profit = 10, 
            min_price_move_from_last_trade = 0.005,
            num_shares_per_trade = 10000,
            training_end_date = None,
            std_dev_influence_on_strategy = None
            ):
        """
        @prices: pandas DataFrame with the prices of the trading instruments
        @trading_instrument_symbol: the symbol of the trading instrument
        @sma_period: the period for the simple moving average
        @std_dev_period: the period for the standard deviation
        @price_dev_num_prices: the number of prices to consider for the price deviation
        @value_for_entry: the value for entry into a position
        @value_to_take_profit: the value to take profit
        @min_price_move_from_last_trade: the minimum price move from the last trade
        @num_shares_per_trade: the number of shares to trade
        @training_end_date: the end date of the training period
        @std_dev_influence_on_strategy: the influence of the standard deviation on the strategy
        """
        
        self.prices = prices
        self._prices_in_use = None # prices used in the strategy, depending on the start and end date, set in _run_strategy
        
        # setting constants and thresholds
        self.trading_instrument_symbol = trading_instrument_symbol
        
        self._non_training_starting_factor = 10
        self._increment_factor = 50
        self._limit_factor = 5
        
        self.symbols = list(self.prices.columns)
        if self.trading_instrument_symbol not in self.symbols:
            raise ValueError("Trading instrument symbol not in data")
        
        self.sma_period = sma_period
        self.std_dev_influence_on_strategy = std_dev_influence_on_strategy
        self._historical_std_dev_mean_trading_instrument = 0
        self.std_dev_period = std_dev_period
        self.price_dev_num_prices = price_dev_num_prices
        
        self.value_for_entry = value_for_entry
        self._original_value_for_entry = value_for_entry
        
        self.value_to_take_profit = value_to_take_profit
        self._original_value_to_take_profit = value_to_take_profit
        self.value_to_take_profit_increment = value_to_take_profit // self._increment_factor
        self._max_value_to_take_profit = value_to_take_profit * self._limit_factor
        self._min_value_to_take_profit = value_to_take_profit // self._limit_factor
        
        self.min_price_move_from_last_trade = min_price_move_from_last_trade
        self.original_min_price_move_from_last_trade = min_price_move_from_last_trade
        
        self.num_shares_per_trade = num_shares_per_trade

        self._number_of_shares_increment = num_shares_per_trade // self._increment_factor
        self._max_shares_per_trade = num_shares_per_trade * self._limit_factor
        self._min_shares_per_trade = num_shares_per_trade // self._limit_factor
        
        self._risk_violated = False
        self._training_mode = True
        
        # setting order management variables
        
        if (training_end_date is None) or (training_end_date not in self.prices.index):
            training_end_date = self.prices.index[len(self.prices) * 2 // 3]
        self.training_end_date = training_end_date
        self._initialize_data_structures_and_variables()
        
        # risk management variables determined based on historical data and used as an upper bound on the risk during the strategy execution
        self._max_limit_on_weekly_pnl_losses = 0
        self._max_limit_on_monthly_pnl_losses = 0
        self._max_total_weekly_volume = 0
        self._max_limit_holding_time = 0
        self._max_position_limit = 0
        
        # risk management variables determined based on historical data and used as a lower bound on the risk during the strategy execution
        self._min_limit_on_weekly_pnl_losses = 0
        self._min_limit_on_monthly_pnl_losses = 0
        self._min_total_weekly_volume = 0
        self._min_limit_holding_time = 0
        self._min_position_limit = 0
        
        # risk management variables determined based on historical data and updated during the strategy execution in non-training mode
        self._risk_limit_weekly_pnl_losses = 0
        self._risk_limit_monthly_pnl_losses = 0
        self._risk_limit_weekly_total_volume = 0
        self._risk_limit_holding_time = 0
        self._risk_limit_max_position = 0
        
        # increment variables for the update of the risk management variables
        self._risk_increment_weekly_pnl_losses = 0
        self._risk_increment_monthly_pnl_losses = 0
        self._risk_increment_weekly_total_volume = 0
        self._risk_increment_holding_time = 0
        self._risk_increment_max_position = 0
        

    def _initialize_data_structures_and_variables(self):
        """
        Helper function of the constructor and of _reset_all_historical_parameters. Initializes the data structures and variables used in the strategy."""
        
        self.orders = []
        self.positions = []
        self.pnls = []
        
        self._last_buy_price = 0
        self._last_sell_price = 0
        self._position = 0
        self._buy_sum_price_qty = 0
        self._buy_sum_qty = 0
        self._sell_sum_price_qty = 0
        self._sell_sum_qty = 0
        self._open_pnl = 0
        self._closed_pnl = 0
        
        self.final_data = None
        self._has_final_data_been_prepared = False

        # data structures for calculating the moving average in the prices
        self._price_history_sma_for_price_dev = {}
        self._sma_for_price_dev = {symbol: 0 for symbol in self.symbols}
        self._prev_price_for_sma = {symbol: 0 for symbol in self.symbols}
        self._prev_sma_for_price_dev = {symbol: 0 for symbol in self.symbols}
        
        # data structures for calculating the standard deviation in the prices
        self._prev_sma_for_std_dev_prices = {symbol: 0 for symbol in self.symbols}
        self._sma_for_std_dev_prices = {symbol: 0 for symbol in self.symbols}
        self._price_history_std_dev_prices = {}
        self._std_dev_prices = {symbol: 0 for symbol in self.symbols}
        self._std_dev_prices_history = {symbol: [] for symbol in self.symbols}
        self._std_dev_prices_history_trading_instrument = []
        
        
        # data structures for calculating the correlation between the trading instrument and the other symbols
        self._price_deviation_from_sma_history = {}
        
        self._sma_for_price_dev_std_dev = {symbol: 0 for symbol in self.symbols}
        self._prev_sma_for_price_dev_std_dev = {symbol: 0 for symbol in self.symbols}
        self._covariance_with_instrument = {symbol: 0 for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        self._correlation_with_instrument = {symbol: 0 for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        self._std_dev_price_deviation = {symbol: 0 for symbol in self.symbols}
        self._std_dev_price_deviation_history = {symbol: [] for symbol in self.symbols}
        self._correlation_history = {symbol: [] for symbol in self.symbols if symbol != self.trading_instrument_symbol}
                
        # data structures for calculating the difference between the projected and actual delta
        self._difference_between_projected_and_actual_delta = {symbol: 0 for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        
        # declaring variables for the risk management        
        self._holding_times = []
        self._current_holding_time = 0

        self.weekly_pnls = []
        self.monthly_pnls = []
        
        self._last_date_num_shares_changed = 0
        self._num_shares_per_trade_history = []
        
        self._total_weekly_volume = 0
        self._weekly_volume_history = []

        
        # variables to keep track of current historical maximums
        self.historical_max_weekly_pnl_losses = 0
        self.historical_max_monthly_pnl_losses = 0
        self.historical_max_weekly_total_volume = 0
        self.historical_max_holding_time = 0
        self.historical_max_position = 0
        
        # data structures for plotting the results
        self.historical_correlations = {symbol: [] for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        self.historical_single_symbol_signal = {symbol: [] for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        self.historical_trading_signal = []
        self._max_position_allowed_history = []
        
        self.figsize = (12, 8)


    def _reset_all_historical_parameters(self):
        """Helper function of the run_strategy_with_risk_optimization_based_on_historical_data. Resets all the historical parameters to their initial values."""
        
        self._has_final_data_been_prepared = False
        self.final_data = None
        self._initialize_data_structures_and_variables()
        
        
    ##################################################### RUN STRATEGY METHODS #####################################################
    
    
    def _run_strategy(self, start = None, end = None):
        """
        Run the strategy from the start date to the end date by iterating over the rows of the prices data. If the end date is not specified, the strategy is run on the whole dataset."""
        
        self._reset_all_historical_parameters() # reset all the historical parameters before running the strategy
        
        if start is None:
            start = self.prices.index[0]
        if end is None:
            end = self.prices.index[-1]
        
        self._prices_in_use = self.prices.loc[start:end]
        for i in range(len(self._prices_in_use)):
            if self._risk_violated:
                self._close_all_positions(i)
                break
            self._on_price_event(self._prices_in_use.iloc[i])


    def run_strategy_with_risk_optimization_based_on_historical_data(self):
        """
        Run the strategy with risk optimization based on the historical data. The strategy is run in training mode up until the @training_end_date. The pnl up until then is plotted.
        The historical risk parameters are determined based on the historical data. The data structures are then reset and the strategy is run in non-training mode with the updated parameters."""
        
        self._run_strategy(start = self.prices.index[0], end = self.training_end_date)
        self._determine_historical_risk_parameters()
        self._training_mode = False
        self.plot_pnls()
        self._run_strategy(start = self.training_end_date, end = self.prices.index[-1])
        self.plot_pnls()

    
    def run_strategy_without_risk_optimization_based_on_historical_data(self):
        """
        Run the strategy on the whole dataset with the parameters that were set by the user."""
        
        self._run_strategy(start = self.training_end_date, end = self.prices.index[-1])

    
    ##################################################### HELPER FUNCTIONS OF THE RUN STRATEGY METHODS #####################################################


    def _on_price_event(self, price_event):
        """Helper function of the _run_strategy method. Updates the data structures and variables based on the current price event. The price event is a row of the price data.
        Enters positions based on the calculated signals. Computes the PnL at the current stage."""
        
        self._store_price_and_update_data_structures(price_event)
        self._create_and_manage_position(price_event)
        self._calculate_and_store_pnl(price_event)


    def _store_price_and_update_data_structures(self, price_event):
        """
        Store the price event and update the data structures used in the strategy. Does the following:
        1. Update the simple moving average for the price deviation
        2. Update the standard deviation in the prices
        3. Update the price deviation from the simple moving average
        4. Update the correlations
        5. Update the difference between the projected and actual delta
        6. Update the threshold constants
        7. Check if the risks were violated
        """
        
        self._update_sma_for_price_dev(price_event)
        self._update_price_deviation_from_sma(price_event)
        self._update_correlations()
        self._update_std_dev_price_deviations(price_event)
        self._update_difference_between_projected_and_actual_delta()
        if self.std_dev_influence_on_strategy is not None and self.std_dev_influence_on_strategy != 0:
            self._update_threshold_constants_based_on_std_dev()
        if not self._training_mode:
            self._update_threshold_constants_based_on_prev_pnl()
            self._check_if_risks_were_violated()


    def _create_and_manage_position(self, price_event):
        """Helper function of the _on_price_event method. Creates and manages the position based on the calculated signals."""
        
        if ((self._difference_between_projected_and_actual_delta > self.value_for_entry and abs(price_event[self.trading_instrument_symbol] - self._last_buy_price) > self.min_price_move_from_last_trade) 
            or (self._position < 0 and self._open_pnl > self.value_to_take_profit)):
            self._execute_trade(1, price_event)
        elif ((self._difference_between_projected_and_actual_delta < -self.value_for_entry and abs(price_event[self.trading_instrument_symbol] - self._last_sell_price) > self.min_price_move_from_last_trade) 
            or (self._position > 0 and self._open_pnl > self.value_to_take_profit)):
            self._execute_trade(-1, price_event)
        else:
            self._execute_trade(0, price_event)


    def _calculate_and_store_pnl(self, price_event):
        """
        Helper function of _on_price_event_method. Calculates the PnL based on the difference between the buy and sell prices and the position. Updates the closed and open PnLs. Appends the PnL to the list of PnLs.
        
        @price_event: current row of the price data
        """
        prev_position = 0
        try:
            prev_position = self.positions[-2]
        except IndexError:
            pass
        
        self._open_pnl = 0
        if self._position > 0:
            self._current_holding_time += 1
            if prev_position < 0:
                # update the closed pnl
                new_buy_sum_price_qty = self._last_buy_price * abs(self._position)
                self._closed_pnl += self._sell_sum_price_qty - (self._buy_sum_price_qty - new_buy_sum_price_qty)
                # update new buy sum price quantity and reset sell quantities
                self._buy_sum_price_qty = new_buy_sum_price_qty
                self._buy_sum_qty = self._position
                self._sell_sum_price_qty = 0
                self._sell_sum_qty = 0
                self._last_sell_price = 0

                self._reset_holding_time()
            elif self._sell_sum_qty > 0:
                # we are long, but we have sold part of our position and have not fully closed it
                assert self._buy_sum_qty > 0
                self._open_pnl = abs(self._sell_sum_qty) * (self._sell_sum_price_qty / self._sell_sum_qty - self._buy_sum_price_qty / self._buy_sum_qty)
                self._open_pnl += abs(self._sell_sum_qty - self._position) * (price_event[self.trading_instrument_symbol] - self._buy_sum_price_qty / self._buy_sum_qty)
        elif self._position < 0:
            self._current_holding_time += 1
            if prev_position > 0:
                # update the closed pnl
                new_sell_sum_price_qty = self._last_sell_price * abs(self._position)
                self._closed_pnl += (self._sell_sum_price_qty - new_sell_sum_price_qty) - self._buy_sum_price_qty
                # update new sell sum price quantity and reset buy quantities
                self._sell_sum_price_qty = new_sell_sum_price_qty
                self._sell_sum_qty = abs(self._position)
                self._buy_sum_price_qty = 0
                self._buy_sum_qty = 0
                self._last_buy_price = 0

                self._reset_holding_time()
            elif self._buy_sum_qty > 0:
                # we are short, but we have bought part of our position and have not fully closed it
                assert self._sell_sum_qty > 0
                self._open_pnl = abs(self._buy_sum_qty) * (self._sell_sum_price_qty / self._sell_sum_qty - self._buy_sum_price_qty / self._buy_sum_qty)
                self._open_pnl += abs(self._buy_sum_qty - self._position) * (self._sell_sum_price_qty / self._sell_sum_qty - price_event[self.trading_instrument_symbol])
        else: # self._position == 0
            self._closed_pnl += (self._sell_sum_price_qty - self._buy_sum_price_qty)
            self._buy_sum_price_qty = 0
            self._buy_sum_qty = 0
            self._sell_sum_price_qty = 0
            self._sell_sum_qty = 0
            self._last_buy_price = 0
            self._last_sell_price = 0
            
            self._reset_holding_time()

        self.pnls.append(self._closed_pnl + self._open_pnl)
        
        self._compute_weekly_and_monthly_pnls()


    ##################################################### HELPER FUNCTIONS OF THE _store_price_and_update_data_structures METHOD #####################################################


    # def _update_std_dev_prices(self, price_event):
    #     """Helper function of the _store_price_and_update_data_structures method. Updates the standard deviation in the prices."""
        
    #     for symbol in self.symbols:
    #         if symbol not in self._price_history_std_dev_prices:
    #                 self._price_history_std_dev_prices[symbol] = deque(maxlen = self.std_dev_period + 1)

    #         self._price_history_std_dev_prices[symbol].append(price_event[symbol])
    #         if len(self._price_history_std_dev_prices[symbol]) < 2:
    #             self._std_dev_prices[symbol] = 0
    #             self._sma_for_std_dev_prices[symbol] = price_event[symbol]
    #         elif len(self._price_history_std_dev_prices[symbol]) > self.std_dev_period:
    #             old_price = self._price_history_std_dev_prices[symbol].popleft()

    #             self._prev_sma_for_std_dev_prices[symbol] = self._sma_for_std_dev_prices[symbol]
    #             self._sma_for_std_dev_prices[symbol] = compute_rolling_mean_online(
    #                 self._sma_for_std_dev_prices[symbol],
    #                 self.std_dev_period,
    #                 len(self._price_history_std_dev_prices[symbol]),
    #                 price_event[symbol],
    #                 old_price
    #                 )

    #             self._std_dev_prices[symbol] = compute_rolling_std_dev_online(
    #                 self._std_dev_prices[symbol],
    #                 self._sma_for_std_dev_prices[symbol],
    #                 self._prev_sma_for_std_dev_prices[symbol],
    #                 self.std_dev_period,
    #                 len(self._price_history_std_dev_prices[symbol]),
    #                 price_event[symbol], 
    #                 old_price
    #                 )
    #         else:
    #             self._prev_sma_for_std_dev_prices[symbol] = self._sma_for_std_dev_prices[symbol]
    #             self._sma_for_std_dev_prices[symbol] = compute_rolling_mean_online(
    #                 self._sma_for_std_dev_prices[symbol],
    #                 self.std_dev_period,
    #                 len(self._price_history_std_dev_prices[symbol]),
    #                 price_event[symbol]
    #                 )
    #             self._std_dev_prices[symbol] = compute_rolling_std_dev_online(
    #                 self._std_dev_prices[symbol],
    #                 self._sma_for_std_dev_prices[symbol],
    #                 self._prev_sma_for_std_dev_prices[symbol],
    #                 self.std_dev_period,
    #                 len(self._price_history_std_dev_prices[symbol]),
    #                 price_event[symbol]
    #                 )
            
    #         if symbol == self.trading_instrument_symbol:
    #             self._std_dev_prices_history_trading_instrument.append(self._std_dev_prices[symbol])
                
    
    def _update_std_dev_price_deviations(self, price_event):
        for symbol in self.symbols:
            if symbol not in self._price_deviation_from_sma_history:
                self._price_deviation_from_sma_history[symbol] = deque(maxlen = self.std_dev_period + 1)
                self._std_dev_price_deviation_history[symbol] = deque(maxlen = self.std_dev_period + 1) 

            self._price_deviation_from_sma_history[symbol].append(price_event[symbol])
            if len(self._price_deviation_from_sma_history[symbol]) < 2:
                self._sma_for_price_dev_std_dev[symbol] = self._price_deviation_from_sma_history[symbol][-1]
                self._std_dev_price_deviation[symbol] = 0
                continue
            
            self._prev_sma_for_price_dev_std_dev[symbol] = self._sma_for_price_dev_std_dev[symbol]     
            if len(self._price_deviation_from_sma_history[symbol]) > self.price_dev_num_prices:
                old_price_dev = self._price_deviation_from_sma_history[symbol].popleft()
                self._update_price_deviation_std_dev(symbol, old_price_dev)
            else:
                self._update_price_deviation_std_dev(symbol)
                
            self._std_dev_price_deviation_history[symbol].append(self._std_dev_price_deviation[symbol])
            
            
    def _update_sma_for_price_dev(self, price_event):
        """Helper function of the _store_price_and_update_data_structures method. Updates the simple moving average for the price deviation. Also stores information in the price histories"""
        for symbol in self.symbols:
            if symbol not in self._price_history_sma_for_price_dev:
                self._price_history_sma_for_price_dev[symbol] = deque(maxlen = self.sma_period + 1)
                self._price_history_std_dev_prices[symbol] = deque(maxlen = self.sma_period + 1)
                
            self._prev_sma_for_price_dev[symbol] = self._sma_for_price_dev[symbol]            
            self._price_history_sma_for_price_dev[symbol].append(price_event[symbol])
            
            self._price_history_std_dev_prices[symbol].append(price_event[symbol])
            
            if len(self._price_history_sma_for_price_dev[symbol]) < 2:
                self._sma_for_price_dev[symbol] = price_event[symbol]
            elif len(self._price_history_sma_for_price_dev[symbol]) > self.sma_period:
                self._prev_price_for_sma[symbol] = self._price_history_sma_for_price_dev[symbol].popleft()
                self._sma_for_price_dev[symbol] = compute_rolling_mean_online(
                    self._sma_for_price_dev[symbol],
                    self.sma_period,
                    len(self._price_history_sma_for_price_dev[symbol]),
                    price_event[symbol],
                    self._prev_price_for_sma[symbol]
                    )
            else:
                n = len(self._price_history_sma_for_price_dev[symbol])
                self._sma_for_price_dev[symbol] = ((n - 1) * self._sma_for_price_dev[symbol] + price_event[symbol]) / n


    def _update_price_deviation_from_sma(self, price_event):
        """Helper function of the _store_price_and_update_data_structures method. Updates the price deviation from the simple moving average."""
        
        for symbol in self.symbols:
            if symbol not in self._price_deviation_from_sma_history:
                self._price_deviation_from_sma_history[symbol] = deque(maxlen = self.price_dev_num_prices)
            self._price_deviation_from_sma_history[symbol].append(price_event[symbol] - self._sma_for_price_dev[symbol])


    def _update_correlations(self):
        """Helper function of the _store_price_and_update_data_structures method. Updates the correlations between the trading instrument and the other symbols."""
        
        old_price_trading_instrument = self._update_sma_std_dev_trading_instrument()
        for symbol in self.symbols:
            if symbol == self.trading_instrument_symbol:
                continue
            
            if len(self._price_history_std_dev_prices[symbol]) < 2:
                self._sma_for_std_dev_prices[symbol] = self._price_history_std_dev_prices[symbol][-1]
                self._std_dev_prices[symbol] = 0
                self._covariance_with_instrument[symbol] = 0
                self._correlation_with_instrument[symbol] = 0
                self.historical_correlations[symbol].append(0)
                continue
            
            new_covariance = 0
            self._prev_sma_for_std_dev_prices[symbol] = self._sma_for_price_dev[symbol]     
            if len(self._price_history_std_dev_prices[symbol]) > self.std_dev_period:
                old_price = self._price_history_std_dev_prices[symbol].popleft()
                self._update_std_dev_prices(symbol, old_price)

                new_covariance = compute_rolling_covariance_online(
                    self._covariance_with_instrument[symbol],
                    self._prev_sma_for_std_dev_prices[symbol],
                    self._prev_sma_for_std_dev_prices[self.trading_instrument_symbol],
                    self.std_dev_period,
                    len(self._price_history_std_dev_prices[symbol]),
                    self._price_history_std_dev_prices[symbol][-1], 
                    self._price_history_std_dev_prices[self.trading_instrument_symbol][-1],
                    old_price,
                    old_price_trading_instrument
                    )
            else:
                self._update_price_deviation_std_dev(symbol)
                new_covariance = compute_rolling_covariance_online(
                    self._covariance_with_instrument[symbol],
                    self._prev_sma_for_std_dev_prices[symbol],
                    self._prev_sma_for_std_dev_prices[self.trading_instrument_symbol],
                    self.std_dev_period,
                    len(self._price_history_std_dev_prices[symbol]),
                    self._price_history_std_dev_prices[symbol][-1], 
                    self._price_history_std_dev_prices[self.trading_instrument_symbol][-1],
                    )
            
            self._std_dev_prices_history[symbol].append(self._std_dev_prices[symbol])
            self._covariance_with_instrument[symbol] = new_covariance
            if self._std_dev_prices[symbol] * self._std_dev_prices[self.trading_instrument_symbol] != 0:
                self._correlation_with_instrument[symbol] = new_covariance / (self._std_dev_prices[symbol] * self._std_dev_prices[self.trading_instrument_symbol])
            else:
                self._correlation_with_instrument[symbol] = 0
                
            # storing for plotting
            if len(self._price_history_std_dev_prices[symbol]) < 10:
                self.historical_correlations[symbol].append(0)
            else:
                self.historical_correlations[symbol].append(self._correlation_with_instrument[symbol])


    def _update_difference_between_projected_and_actual_delta(self):
        """Helper function of the _store_price_and_update_data_structures method. Updates the difference between the projected, as predicted by the leading instruments, and actual delta observed in reality."""
        
        difference_actual_and_projected_price_deviation_based_on_other_symbol = {symbol:0 for symbol in self.symbols if symbol != self.trading_instrument_symbol}
        for symbol in self.symbols:
            if symbol == self.trading_instrument_symbol:
                continue
            
            projected_price_deviation_based_on_other_symbol = self._covariance_with_instrument[symbol] * self._price_deviation_from_sma_history[symbol][-1]
            difference_actual_and_projected_price_deviation_based_on_other_symbol[symbol] = projected_price_deviation_based_on_other_symbol - self._price_deviation_from_sma_history[self.trading_instrument_symbol][-1]
            # storing for plotting
            self.historical_single_symbol_signal[symbol].append(difference_actual_and_projected_price_deviation_based_on_other_symbol[symbol])
        total_correlation = sum([abs(corr) for corr in self._correlation_with_instrument.values()])
        total_weighted_difference_with_projected_price_deviation = 0
        if total_correlation != 0:   
            total_weighted_difference_with_projected_price_deviation = sum([difference_actual_and_projected_price_deviation_based_on_other_symbol[symbol] * abs(self._correlation_with_instrument[symbol]) 
                                                                            for symbol in self.symbols if symbol != self.trading_instrument_symbol]) / total_correlation

        self._difference_between_projected_and_actual_delta = total_weighted_difference_with_projected_price_deviation
        # storing for plotting
        self.historical_trading_signal.append(total_weighted_difference_with_projected_price_deviation)


    def _update_threshold_constants_based_on_std_dev(self):
        """Helper function of the _store_price_and_update_data_structures method. Updates the threshold constants based on the standard deviation of the price deviation from the simple moving average 
        if @_std_dev_influence_on_strategy was not set to 0. In non-training mode, the thresholds are updated based on the historical mean of the standard deviation of the price deviation from the simple moving average."""
        
        assert self.std_dev_influence_on_strategy is not None and self.std_dev_influence_on_strategy != 0
        if self._training_mode:
            std_dev_factor = self._std_dev_price_deviation[self.trading_instrument_symbol] / self.std_dev_influence_on_strategy
        else:
            std_dev_factor = self._historical_std_dev_mean_trading_instrument
        self.value_for_entry = self._original_value_for_entry / std_dev_factor
        self.value_to_take_profit = self._original_value_to_take_profit / std_dev_factor
        self.min_price_move_from_last_trade = self.original_min_price_move_from_last_trade * std_dev_factor


    def _update_threshold_constants_based_on_prev_pnl(self):
        """Helper function of the _store_price_and_update_data_structures method. Updates the threshold constants based on the PnL of the previous week. Only works in non-training mode, when all the thresholds have been determined using historical data."""
        
        n = len(self.pnls) - self._last_date_num_shares_changed
        if n >= 5 and n % 5 == 0:
            if self.weekly_pnls[-1] > 0:
                self._last_date_num_shares_changed = len(self.pnls)
                self._increment_num_shares_per_trade()
                self._increment_value_to_take_profit()
                self._decrement_weekly_pnl_losses_risk_limit()
                self._decrement_monthly_pnl_losses_risk_limit()
                self._increment_weekly_total_volume_risk_limit()
                self._increment_max_position_allowed()      
            elif self.weekly_pnls[-1] < 0:
                self._last_date_num_shares_changed = len(self.pnls)
                self._decrement_num_shares_per_trade()
                self._decrement_value_to_take_profit()
                self._increment_weekly_pnl_losses_risk_limit()
                self._increment_monthly_pnl_losses_risk_limit()
                self._decrement_weekly_total_volume_risk_limit()
                self._decrement_max_position_allowed()
                

            self._num_shares_per_trade_history.append(self.num_shares_per_trade)
            self._max_position_allowed_history.append(self._risk_limit_max_position)


    def _check_if_risks_were_violated(self):
        """Helper function of _store_price_and_update_data_structures. Checks if the risks were violated based on the historical data and the current data. If the risks were violated, the strategy is stopped."""
        
        if len(self.weekly_pnls) > 0 and self.weekly_pnls[-1] < self._risk_limit_weekly_pnl_losses:
            self._risk_violated = True
            print("Risk violated: weekly pnl losses exceeded the limit")
        if len(self.monthly_pnls) > 0 and self.monthly_pnls[-1] < self._risk_limit_monthly_pnl_losses:
            self._risk_violated = True
            print("Risk violated: monthly pnl losses exceeded the limit")
        if len(self._weekly_volume_history) > 0 and self._weekly_volume_history[-1] > self._risk_limit_weekly_total_volume:
            self._risk_violated = True
            print("Risk violated: weekly total volume exceeded the limit")
        if len(self._holding_times) > 0 and self._holding_times[-1] > self._risk_limit_holding_time:
            self._risk_violated = True
            print("Risk violated: holding time exceeded the limit")

            
    ##################################################### SECOND LEVEL HELPER FUNCTIONS OF THE _store_price_and_update_data_structures METHOD #####################################################    
    
    def _update_price_deviation_std_dev(self, symbol, old_price_dev = None):
        """Helper function of the _update_correlations method. Updates the standard deviation of the price deviation from the simple moving average."""
        
        self._sma_for_price_dev_std_dev[symbol] = compute_rolling_mean_online(
            self._sma_for_price_dev_std_dev[symbol],
            self.price_dev_num_prices,
            len(self._price_deviation_from_sma_history[symbol]),
            self._price_deviation_from_sma_history[symbol][-1],
            old_price_dev
            )
        
        self._std_dev_price_deviation[symbol] = compute_rolling_std_dev_online(
            self._std_dev_price_deviation[symbol],
            self._sma_for_price_dev_std_dev[symbol],
            self._prev_sma_for_price_dev_std_dev[symbol],
            self.price_dev_num_prices,
            len(self._price_deviation_from_sma_history[symbol]),
            self._price_deviation_from_sma_history[symbol][-1],
            old_price_dev
            )
        
    
    def _update_std_dev_prices(self, symbol, old_price = None):
        """Helper function of the _update_correlations method. Updates the standard deviation of the price deviation from the simple moving average."""
        
        self._sma_for_std_dev_prices[symbol] = compute_rolling_mean_online(
            self._sma_for_std_dev_prices[symbol],
            self.std_dev_period,
            len(self._price_history_sma_for_price_dev[symbol]),
            self._price_history_sma_for_price_dev[symbol][-1],
            old_price
            )
        
        self._std_dev_prices[symbol] = compute_rolling_std_dev_online(
            self._std_dev_prices[symbol],
            self._sma_for_std_dev_prices[symbol],
            self._prev_sma_for_std_dev_prices[symbol],
            self.price_dev_num_prices,
            len(self._price_history_sma_for_price_dev[symbol]),
            self._price_history_sma_for_price_dev[symbol][-1],
            old_price
            )

    
    def _update_sma_std_dev_trading_instrument(self):
        """Helper function of the _update_correlations method. Updates the simple moving average and the standard deviation of the price deviation from the simple moving average for the trading instrument only."""
        
        if len(self._price_history_std_dev_prices[self.trading_instrument_symbol]) < 2:
            self._sma_for_std_dev_prices[self.trading_instrument_symbol] = self._price_history_std_dev_prices[self.trading_instrument_symbol][-1]
            self._std_dev_prices[self.trading_instrument_symbol] = 0
            self._price_history_std_dev_prices[self.trading_instrument_symbol].append(0)
        elif len(self._price_history_std_dev_prices[self.trading_instrument_symbol]) > self.price_dev_num_prices:
            old_price = self._price_history_std_dev_prices[self.trading_instrument_symbol].popleft()
            self._update_std_dev_prices(self.trading_instrument_symbol, old_price)
            return old_price
        else:
            self._update_std_dev_prices(self.trading_instrument_symbol)
            
        

        return None


    ##################################################### HELPER FUNCTIONS OF THE _create_and_manage_position METHOD #####################################################


    def _execute_trade(self, signal, price_event):
        """Helper function of the _create_and_manage_position method. Executes the trade based on the signal and appends the order to self.orders. If the signal is 0, no trade is executed. If the signal is 1, a buy trade is executed. If the signal is -1, a sell trade is executed."""
        
        if signal == 0:
            self.orders.append(0)
        elif signal == 1:
            self._update_historical_data_on_trade_execution()
            self.orders.append(1)
            self._execute_buy_trade(price_event)
        else:
            self._update_historical_data_on_trade_execution()
            self.orders.append(-1)
            self._execute_sell_trade(price_event)
            
        self._append_position()


    def _update_historical_data_on_trade_execution(self):
        """Helper function of the _execute_trade method. Updates the total_weekly_volume if a trade was executed."""
        
        self._total_weekly_volume += self.num_shares_per_trade
        if len(self.orders) > 0 and len(self.orders) % 5 == 0:
            self._weekly_volume_history.append(self._total_weekly_volume)
            if self._total_weekly_volume > self._max_total_weekly_volume:
                self._max_total_weekly_volume = self._total_weekly_volume
            self._total_weekly_volume = 0


    def _execute_buy_trade(self, price_event):
        """Helper function of the _execute_trade method. Executes a buy trade."""
        
        self._last_buy_price = price_event[self.trading_instrument_symbol]
        self._position += self.num_shares_per_trade
        self._buy_sum_price_qty += price_event[self.trading_instrument_symbol] * self.num_shares_per_trade
        self._buy_sum_qty += self.num_shares_per_trade


    def _execute_sell_trade(self, price_event):
        """Helper function of the _execute_trade method. Executes a sell trade."""
        
        self._last_sell_price = price_event[self.trading_instrument_symbol]
        self._position -= self.num_shares_per_trade
        self._sell_sum_price_qty += price_event[self.trading_instrument_symbol] * self.num_shares_per_trade
        self._sell_sum_qty += self.num_shares_per_trade

    
    def _append_position(self):
        """Helper function of the _execute_buy_trade and _execute_sell_trade methods. Appends the current position to the list of positions and updates the historical maximum position if necessary."""
        
        self.positions.append(self._position)
        if abs(self._position) > self.historical_max_position:
            self.historical_max_position = abs(self._position)


    def _reset_holding_time(self):
        """Helper function of the _calculate_and_store_pnl method. Appends the final holding time to the _holding_times dataset. Resets the current holding time."""
        
        self._holding_times.append(self._current_holding_time)
        if self._current_holding_time > self.historical_max_holding_time:
            self.historical_max_holding_time = self._current_holding_time
        self._current_holding_time = 0


    ##################################################### HELPER FUNCTIONS OF THE _calculate_and_store_pnl METHOD #####################################################


    def _close_all_positions(self, index):
        """Helper function of the _run_strategy method. Closes all positions if one of the risks were violated and updates the pnl accordingly."""
        
        if self._position > 0:
            self._sell_sum_price_qty += self._position * self._prices_in_use[self.trading_instrument_symbol].iloc[index]
            self._sell_sum_qty += self._position
        elif self._position < 0:
            self._buy_sum_price_qty += self._position * self._prices_in_use[self.trading_instrument_symbol].iloc[index]
            self._buy_sum_qty += self._position
        self._position = 0
        self._calculate_and_store_pnl(self._prices_in_use.iloc[index])


    def _compute_weekly_and_monthly_pnls(self):
        """Helper function of _calculate_and_store_pnl. Computes the weekly and monthly PnLs based on the last 5 and 20 PnLs. Updates the historical maximum weekly and monthly PnL losses if necessary."""
        
        n = len(self.pnls)
        if n > 5:
            weekly_pnl = self.pnls[-1] - self.pnls[-6]
            self.weekly_pnls.append(weekly_pnl)
            if weekly_pnl < self.historical_max_weekly_pnl_losses:
                self.historical_max_weekly_pnl_losses = weekly_pnl
        else:
            self.weekly_pnls.append(0)
            
        if n > 20:
            monthly_pnl = self.pnls[-1] - self.pnls[-21]
            self.monthly_pnls.append(monthly_pnl)
            if monthly_pnl < self.historical_max_monthly_pnl_losses:
                self.historical_max_monthly_pnl_losses = monthly_pnl
        else:
            self.monthly_pnls.append(0)
                

    ##################################################### HELPER FUNCTIONS THAT UPDATE THRESHOLDS #####################################################
    

    def _determine_historical_risk_parameters(self):
        """
        Helper function of the run_strategy_with_risk_optimization_based_on_historical_data method. Determines the historical risk parameters based on the historical data. 
        The historical risk parameters are the maximum weekly and monthly pnl losses, the maximum holding time, the maximum total weekly volume, and the maximum position limit.
        Prepares the thresholds to be updated during the strategy execution in non-training mode, where they are relaxed or become more stringent based on the past weeks' performance.
        """
        
        self._max_limit_on_weekly_pnl_losses = self.historical_max_weekly_pnl_losses * self._limit_factor
        self._max_limit_on_monthly_pnl_losses = self.historical_max_monthly_pnl_losses * self._limit_factor
        self._max_total_weekly_volume = self.historical_max_weekly_total_volume * self._limit_factor
        self._max_limit_holding_time = self.historical_max_holding_time * self._limit_factor
        self._max_position_limit = self.historical_max_position * self._limit_factor
        
        self._min_limit_on_weekly_pnl_losses = self.historical_max_weekly_pnl_losses // self._limit_factor
        self._min_limit_on_monthly_pnl_losses = self.historical_max_monthly_pnl_losses // self._limit_factor
        self._min_total_weekly_volume = self.historical_max_weekly_total_volume // self._limit_factor
        self._min_limit_holding_time = self.historical_max_holding_time // self._limit_factor
        self._min_position_limit = self.historical_max_position // self._limit_factor
        
        self._risk_increment_weekly_pnl_losses = self.historical_max_weekly_pnl_losses // self._increment_factor
        self._risk_increment_monthly_pnl_losses = self.historical_max_monthly_pnl_losses // self._increment_factor
        self._risk_increment_weekly_total_volume = self.historical_max_weekly_total_volume // self._increment_factor
        self._risk_increment_holding_time = self.historical_max_holding_time // self._increment_factor
        self._risk_increment_max_position = self.historical_max_position // self._increment_factor
        
        self.num_shares_per_trade //= self._non_training_starting_factor
        self.value_to_take_profit //= self._non_training_starting_factor
        self._risk_limit_weekly_pnl_losses = self.historical_max_weekly_pnl_losses // self._non_training_starting_factor
        self._risk_limit_monthly_pnl_losses = self.historical_max_monthly_pnl_losses // self._non_training_starting_factor
        self._risk_limit_weekly_total_volume = self.historical_max_weekly_total_volume // self._non_training_starting_factor
        self._risk_limit_holding_time = self.historical_max_holding_time // self._non_training_starting_factor
        self._risk_increment_max_position = self.historical_max_position // self._non_training_starting_factor
        
        self._historical_std_dev_mean_trading_instrument = sum(self._std_dev_prices_history_trading_instrument[self.trading_instrument_symbol]) / len(self._std_dev_prices_history_trading_instrument[self.trading_instrument_symbol])
    

    ##################################################### HELPER FUNCTIONS OF _update_threshold_constants_based_on_prev_pnl #####################################################

    def _increment_weekly_pnl_losses_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the weekly PnL losses risk limit."""
        
        self._risk_limit_weekly_pnl_losses += self._risk_increment_weekly_pnl_losses
        if self._risk_limit_weekly_pnl_losses > self._min_limit_on_weekly_pnl_losses:
            self._risk_limit_weekly_pnl_losses = self._min_limit_on_weekly_pnl_losses
        
    def _decrement_weekly_pnl_losses_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the weekly PnL losses risk limit."""
        
        self._risk_limit_weekly_pnl_losses -= self._risk_increment_weekly_pnl_losses
        if self._risk_limit_weekly_pnl_losses < self._max_limit_on_weekly_pnl_losses:
            self._risk_limit_weekly_pnl_losses = self._max_limit_on_weekly_pnl_losses
    
    def _increment_monthly_pnl_losses_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the monthly PnL losses risk limit."""
        
        self._risk_limit_monthly_pnl_losses += self._risk_increment_monthly_pnl_losses
        if self._risk_limit_monthly_pnl_losses > self._min_limit_on_monthly_pnl_losses:
            self._risk_limit_monthly_pnl_losses = self._min_limit_on_monthly_pnl_losses
    
    def _decrement_monthly_pnl_losses_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the monthly PnL losses risk limit."""
        
        self._risk_limit_monthly_pnl_losses -= self._risk_increment_monthly_pnl_losses
        if self._risk_limit_monthly_pnl_losses < self._max_limit_on_monthly_pnl_losses:
            self._risk_limit_monthly_pnl_losses = self._max_limit_on_monthly_pnl_losses
    
    def _increment_weekly_total_volume_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the weekly total volume risk limit."""
        
        self._risk_limit_weekly_total_volume += self._risk_increment_weekly_total_volume
        if self._risk_limit_weekly_total_volume > self._max_total_weekly_volume:
            self._risk_limit_weekly_total_volume = self._max_total_weekly_volume
            
    def _decrement_weekly_total_volume_risk_limit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the weekly total volume risk limit."""
        self._risk_limit_weekly_total_volume -= self._risk_increment_weekly_total_volume
        if self._risk_limit_weekly_total_volume < self._min_total_weekly_volume:
            self._risk_limit_weekly_total_volume = self._min_total_weekly_volume
    
    def _increment_num_shares_per_trade(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the number of shares per trade."""
        self.num_shares_per_trade += self._number_of_shares_increment
        if self.num_shares_per_trade > self._max_shares_per_trade:
            self.num_shares_per_trade = self._max_shares_per_trade
            
    def _decrement_num_shares_per_trade(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the number of shares per trade."""
        
        self.num_shares_per_trade -= self._number_of_shares_increment
        if self.num_shares_per_trade < self._min_shares_per_trade:
            self.num_shares_per_trade = self._min_shares_per_trade
            
    def _increment_value_to_take_profit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the value to take profit."""
        
        self.value_to_take_profit += self.value_to_take_profit_increment
        if self.value_to_take_profit > self._max_value_to_take_profit:
            self.value_to_take_profit = self._max_value_to_take_profit
            
    def _decrement_value_to_take_profit(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the value to take profit."""
        
        self.value_to_take_profit -= self.value_to_take_profit_increment
        if self.value_to_take_profit < self._min_value_to_take_profit:
            self.value_to_take_profit = self._min_value_to_take_profit
            
    def _increment_max_position_allowed(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Increments the maximum position allowed."""
        
        self._max_position_allowed += self._max_position_increment
        if self._max_position_allowed > self._max_position_limit:
            self._max_position_allowed = self._max_position_limit
        
    def _decrement_max_position_allowed(self):
        """Helper function of the _update_threshold_constants_based_on_prev_pnl method. Decrements the maximum position allowed."""
        
        self._max_position_allowed -= self._max_position_increment
        if self._max_position_allowed < self._min_position_limit:
            self._max_position_allowed = self._min_position_limit


    ##################################################### PLOTTING METHODS #####################################################


    def _prepare_data_for_plotting(self):
        """Helper function of the plotting functions. Prepares the final data for plotting by saving the historical information on the self.final_data dataframe."""
        rolling_sharpes = compute_rolling_sharpe_ratio(self.pnls)
        rolling_sortinos = compute_rolling_sortino_ratio(self.pnls)
        
        self.final_data = pd.DataFrame(
            data = 
            {
                'PnLs': self.pnls,
                'Orders': self.orders,
                'Positions': self.positions,
                'Final_signal': self.historical_trading_signal,
                'Weekly_pnls': self.weekly_pnls,
                'Monthly_pnls': self.monthly_pnls,
                'Rolling_Sharpes': rolling_sharpes,
                'Rolling_Sortinos': rolling_sortinos
            }, 
            index = self._prices_in_use.index)
        for symbol in self.symbols:
            if symbol != self.trading_instrument_symbol:
                self.final_data['Correlation_with_' + symbol] = self.historical_correlations[symbol]
                self.final_data['Signal__with' + symbol] = self.historical_single_symbol_signal[symbol]
        self._has_final_data_been_prepared = True
        
        if not self._training_mode:
            self.final_data["Max_position_allowed"] = self._max_position_allowed_history
            self.final_data['Num_shares_per_trade'] = self._num_shares_per_trade_history,


    def plot_pnls(self):
        """Plots the PnLs."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        index = self.final_data.index
        plt.plot(index, self.pnls, color = 'k', lw = 1., label = 'PnL')
        plt.plot(self.final_data.loc[self.final_data['PnLs'] > 0].index,
                 self.final_data['PnLs'][self.final_data['PnLs'] > 0],
                 color = 'g', lw = 0, marker = '.')
        plt.plot(self.final_data.loc[self.final_data['PnLs'] < 0].index,
                 self.final_data['PnLs'][self.final_data['PnLs'] < 0],
                 color = 'r', lw = 0, marker = '.')
        sharpe_ratio = compute_sharpe_ratio(self.pnls)
        sortino_ratio = compute_sortino_ratio(self.pnls)
        plt.figtext(0.15, 0.8, f'Sharpe Ratio: {sharpe_ratio:.2f}\nSortino Ratio: {sortino_ratio:.2f}', 
                    bbox=dict(facecolor='white', alpha=0.5))
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.title('PnLs')
        plt.show()
        

    def plot_correlations(self):
        """Plots the rolling correlation between the trading instrument and the leading instruments over time."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        for symbol in self.symbols:
            if symbol != self.trading_instrument_symbol:
                plt.plot(self.final_data.index, self.final_data['Correlation_with_' + symbol], label = 'Correlation with ' + symbol)

        for i in np.arange(-1, 1, 0.25):
            plt.axhline(y = i, color = 'k', lw = 0.5, ls = '--')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.title('Correlations')
        plt.legend()
        plt.show()
        
    
    def plot_single_signals(self):
        """Plots the predicted signals if only one of the leading instruments was used."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        for symbol in self.symbols:
            if symbol != self.trading_instrument_symbol:
                plt.plot(self.final_data.index, self.final_data['Signal_with_' + symbol], label = 'Signal with ' + symbol)
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Signal')
        plt.title('How single signals influence the trading signal')
        plt.legend()
        plt.show()
        
    
    def plot_trading_signal(self):
        """Plots the trading signal with the thresholds for entry and exit and the buy and sell orders."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.plot(self.final_data.index, self.final_data['Final_signal'], label = 'Final signal')
        plt.plot(self.final_data.loc[self.final_data['Orders'] == 1].index,
                 self.final_data['Final_signal'].loc[self.final_data['Orders'] == 1],
                 color = 'g', lw = 0, marker = '^', markersize = 7, label = 'Buy')
        plt.plot(self.final_data.loc[self.final_data['Orders'] == -1].index,
                 self.final_data['Final_signal'].loc[self.final_data['Orders'] == -1],
                 color = 'r', lw = 0, marker = 'v', markersize = 7, label = 'Sell')
        for i in np.arange(self.value_for_entry, self.value_for_entry * 10, self.value_for_entry * 2):
            plt.axhline(y = i, color = 'g', lw = 0.5, ls = '--')
        for i in np.arange(-self.value_for_entry, -self.value_for_entry * 10, -self.value_for_entry * 2):
            plt.axhline(y = i, color = 'r', lw = 0.5, ls = '--')
        plt.axhline(y = 0, color = 'k', lw = 0.5, ls = '--')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Signal')
        plt.title('Trading signal')
        plt.legend()
        plt.show()
        
    
    def plot_orders_over_prices(self):
        """Plots the orders over the price of the trading instrument."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.plot(self.final_data.index, self._prices_in_use[self.trading_instrument_symbol], label = 'Price')
        plt.plot(self.final_data.loc[self.final_data['Orders'] == 1].index,
                    self._prices_in_use[self.trading_instrument_symbol].loc[self.final_data['Orders'] == 1],
                    color = 'g', lw = 0, marker = '^', markersize = 7, label = 'Buy')
        plt.plot(self.final_data.loc[self.final_data['Orders'] == -1].index,
                    self._prices_in_use[self.trading_instrument_symbol].loc[self.final_data['Orders'] == -1],
                    color = 'r', lw = 0, marker = 'v', markersize = 7, label = 'Sell')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Orders over prices')
        plt.legend()
        plt.show()
        
    
    def plot_positions(self):
        """Plots the position of the strategy over time."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.plot(self.final_data.index, self.final_data['Positions'], label = 'Positions')
        if not self._training_mode:
            plt.plot(self.final_data.index, self._max_position_allowed_history, color = 'k', lw = 0.5, ls = '--', label = 'Max position allowed')
            plt.plot(self.final_data.index, -self._max_position_allowed_history, color = 'k', lw = 0.5, ls = '--', label = 'Min position allowed')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Position')
        plt.title('Positions')
        plt.legend()
        plt.show()
        
    
    def plot_weekly_pnl_distribution(self):
        """Plots the distribution of the weekly PnLs."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.hist(self.weekly_pnls, bins = 20, color = 'b', edgecolor = 'k')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.title('Weekly PnL distribution')
        plt.show()
        

    def plot_position_distribution(self):
        """Plots the distribution of the positions."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.hist(self.positions, bins = 20, color = 'b', edgecolor = 'k')
        plt.xlabel('Position')
        plt.ylabel('Frequency')
        plt.title('Positions distribution')
        plt.show()
    
    
    def plot_holding_times_distribution(self):
        """§Plots the distribution of the holding times."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        holding_times_str = 'Holding times'
        plt.figure(figsize = self.figsize)
        plt.hist(self._holding_times, label = holding_times_str, bins=20, color = 'b', edgecolor = 'k')
        plt.xlabel(holding_times_str)
        plt.ylabel('Frequency')
        plt.title(holding_times_str)
        plt.legend()
        plt.show()

    
    def plot_weekly_volume_distribution(self):
        """Plots the distribution of the weekly total volume."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.hist(self._weekly_volume_history, bins = 20, color = 'b', edgecolor = 'k')
        plt.xlabel('Weekly total volume')
        plt.ylabel('Frequency')
        plt.title('Weekly total volume distribution')
        plt.show()
    

    def plot_max_drawdown(self):
        """Computes the maximum drawdown and plots the PnLs indicating where the maximum drawdown occurred."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        max_drawdown, drawdown_max_index, drawdown_min_index = compute_max_drawdown(self.pnls)
        plt.figure(figsize = self.figsize)
        plt.plot(self.final_data.index, self.final_data['PnLs'], label = 'PnL')
        plt.axhline(y = self.pnls[drawdown_max_index], color = 'g', lw = 0.5, ls = '--')
        plt.axhline(y = self.pnls[drawdown_min_index], color = 'r', lw = 0.5, ls = '--')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.title(f'Max drawdown: {max_drawdown}.2f')
        plt.legend()
        plt.show()
        return max_drawdown
    
    
    def plot_rolling_sharpes_and_sortino_ratios(self):
        """Plots the rolling Sharpe and Sortino ratios over time."""
        
        if not self._has_final_data_been_prepared:
            self._prepare_data_for_plotting()
        plt.figure(figsize = self.figsize)
        plt.plot(self.final_data.index, self.final_data['Rolling_Sharpes'], label = 'Rolling Sharpe ratio')
        plt.plot(self.final_data.index, self.final_data['Rolling_Sortinos'], label = 'Rolling Sortino ratio')
        plt.xticks([self.final_data.index[i] for i in range(len(self.final_data)) if i % (len(self.final_data) // 8) == 0])
        plt.xlabel('Date')
        plt.ylabel('Ratio')
        plt.title('Rolling Sharpe and Sortino ratios')
        plt.legend()
        plt.show()
        