import numpy as np
import math

def sma(data):
    """Simple Moving Average
    :arg
        data: array of the data, len(data) = period of the SMA
    :returns
        array of simple moving averages
    """
    ma = 0
    period = len(data)
    for i in range(period):
        ma += data[i]
    return ma / period
    
    
def rsi(data):
    """Relative Strength Index
    :arg
        data: array of the data from which to calculate RSI, len(data) = period for the RSI
    :returns
        array of the custom RSI values (values in range [0, 1])
    """
    period_rsi = len(data)
    num_vars = len(data[0])
    rsi_values = []
    rsi_val = 0
    
    for j in range(num_vars):
        avg_gain = 0
        avg_loss = 0
        num_g = 0
        num_l = 0
        
        for i in range(period_rsi):
            if data[i][j] >= 0:
                avg_gain += data[i][j]
                num_g += 1
            else:
                avg_loss += abs(data[i][j])
                num_l += 1
        if avg_loss == 0:
            rsi_values.append(1)
        elif avg_gain == 0:
            rsi_values.append(0)
        else:
            # classic RSI
            rsi_val = 100 - 100 / (1 + (avg_gain/num_g) / (avg_loss/num_l))
            # custom transformation: exaggerate values in (0, 30) u (70, 100) and project to (0, 1)
            rsi_val = 1 / (1 + np.exp(0.4 * (rsi_val - 30))) + 1 / (1 + np.exp(0.4 * (rsi_val - 70))) - 1
            rsi_values.append(rsi_val)
        
    return np.array(rsi_values)


def mae(data_point, ma_exponential, period):
    """Moving Average Exponential
    :arg
        data_point: current data point
        ma_exponential: previous mae value
        period: the period for moving average
    :returns
        array of exponential moving averages
    """
    alpha = 2 / (period + 1)
    return alpha * data_point + (1 - alpha) * ma_exponential # returns moving average exponential
    

def stoch_osc(data):
    """Stochastic Oscillator
    :arg
        data: array of the data from which to calculate Stochastic Oscillator, len(data) = period
        custom stochastic oscillator
    """
    num_vars = len(data[0])
    period = len(data)
    low = [0] * num_vars
    high = [0] * num_vars
    
    osc_values = []
    for j in range(num_vars):
        for i in range(period):
            if data[i][j] > high[j]:
                high[j] = data[i][j]
                
            if data[i][j] < low[j]:
                low[j] = data[i][j]
        osc_values.append((abs(data[-1][j] - low[j])) / (abs(high[j] - low[j])))
        
    return osc_values # returns value from range [0, 1]


def vol_osc(volume_period1, volume_period2):
    """Volume Oscillator
    :arg
        volume_period1: log-volume data with the period of short term simple moving average
        volume_period2: log_volume data with the period of long term simple moving average
    :returns
        volume oscillator
    """
    short_sma = sma(volume_period1)
    long_sma = sma(volume_period2)
    if np.any(short_sma == 0):
        return short_sma
    return (short_sma - long_sma) / short_sma # returns value from range [0, 1]


def build_regressor(r, v, p):
    """Function to construct array that contains regressors
    Contains: previous y_t, log-volume, short term SMA of volume and previous y_t,
    inflation data, VIX, sin with different frequencies, volume oscillator, stochastic oscillator,
    MACD, and modified RSI
    :arg
        r: array of previous returns
        v: array of log-volume data
        p: AR lag
    :returns
        array of regressors X
    """
    rsi_values = []
    lag_values = []
    mae_short = np.zeros(len(r[0]))
    short_period = 5
    long_period = 10
    mae_long = np.zeros(len(r[0]))
    macd_values = []
    stoch_osc_values = []
    vol_osc_values = []
    lag_volumes = []
    for j in range(p+1, len(r)):
        mae_short = mae(r[j-1], mae_short, short_period)
        mae_long = mae(r[j-1], mae_long, long_period)
        macd_values.append(mae_long - mae_short)
        stoch_osc_values.append(stoch_osc(r[j-p:j]))
        vol_osc_values.append(vol_osc(v[j-p:j], v[j-1]))
        rsi_values.append(rsi(r[j-p:j])) # RSI from and including r[j-1] (p+1 observations)
        lag_values.append(np.array(r[j-p:j]).flatten()) # arrays of lagged values up to and including r[j-1]
        lag_volumes.append(np.array(v[j-p:j]).flatten())

    x = np.hstack([lag_values, macd_values, stoch_osc_values, rsi_values, np.ones((np.asarray(lag_values).shape[0], 1))])
    return x

def build_reduced_regressor(r, v, p):
    """To make the work-flow simple. Call indicators.build_regressor() to obtain the full regressors array.
    Then run structure estimation, simply comment out the insignificant regressors and adjust the autoregression
    depth for returns and volumes
    :arg
        r: array of previous returns
        v: array of log-volume data
        p: AR lag
    :returns
        array of (reduced) regressors X
    """
    rsi_values = []
    lag_values = []
    mae_short = np.zeros(len(r[0]))
    short_period = 5
    long_period = 10
    mae_long = np.zeros(len(r[0]))
    macd_values = []
    stoch_osc_values = []
    vol_osc_values = []
    lag_volumes = []
    for j in range(p+1, len(r)):
        mae_short = mae(r[j-1], mae_short, short_period)
        mae_long = mae(r[j-1], mae_long, long_period)
        macd_values.append(mae_long - mae_short)
        stoch_osc_values.append(stoch_osc(r[j - p:j]))
        vol_osc_values.append(vol_osc(v[j - p:j], v[j-1]))
        rsi_values.append(rsi(r[j - p:j]))  # RSI from and including r[j-1] (p+1 observations)
        lag_values.append(np.array(r[j - p:j]).flatten())  # arrays of lagged values up to and including r[j-1]
        lag_volumes.append(np.array(v[j - 1:j]).flatten())

    x = np.hstack([lag_values, macd_values, np.ones((np.asarray(lag_values).shape[0], 1))])
    return x