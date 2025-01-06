import numpy as np
import math

def sma(t, data, period):
    """
    Args:
        t: current time step
        data: data array
        period: the period for moving average
    Returns:
        tuple of simple moving averages
    """
    ma = 0
    for i in range(period):
        ma += data[t-i]
    return ma / period
    
    
def rsi(t, data, period):
    """
    Args:
        t: current time step
        data: data array
    Returns:
        tuple of the custom transformation of the RSI indicator from range [0, 1]
    """
    num_vars = len(data[t])
    
    rsi_values = []
    
    for j in range(num_vars):
        avg_gain = 0
        avg_loss = 0
        num_g = 0
        num_l = 0
        
        for i in range(period):
            if data[t-i][j] >= 0:
                avg_gain += data[t-i][j]
                num_g += 1
            else:
                avg_loss += abs(data[t-i][j])
                num_l += 1
        if avg_loss == 0:
            rsi_values.append(1)
        elif avg_gain == 0:
            rsi_values.append(0)
        else:
            rsi = 100 - 100 / (1 + (avg_gain/num_g) / (avg_loss/num_l)) # classic RSI
            rsi = 1 / (1 + np.exp(0.4 * (rsi - 30))) + 1 / (1 + np.exp(0.4 * (rsi - 70))) - 1
            rsi_values.append(rsi)
        
    return np.array(rsi_values)


def mae(data_point, ma_exponential, period):
    """
    Args:
        data_point: current data point
        ma_exponential: previous mae value
        period: the period for moving average
    Returns:
        tuple of exponential moving averages
    """
    alpha = 2 / (period + 1)
    return alpha * data_point + (1 - alpha) * ma_exponential # returns moving average exponential


def mom_osc(t, data, period):
    """
    Args:
        t: current time step
        data: data array
        period: the period for moving average
    Returns:
        momentum oscilator
    """
    num_g, num_l = 0
    for i in range(period):
        if data[t-i] >= 0:
            num_g += 1
        else:
            num_l += 1
    return (num_g - num_l) / (num_g + num_l) # range 0, 1 that indicates the current momentum
        

def macd(mae1, mae2):
    """
    Args:
        mae1: short term moving average exponential
        mae2: long term moving average exponential
    Returns:
        moving average convergence divergence
    """
    return mae1 - mae2 # MACD simply returns the difference between short and long term exponential moving averages (I am using 21 and 7 period mae)
    

def stoch_osc(t, data, period):
    """
    Args:
        t: current time step
        data: data array
        period: the period for moving average
    Returns:
        custom stochastic oscilator
    """
    low, high = 0
    for i in range(period):
        if data[t-i] > high:
            high = data[t-i]
        if data[t-i] < low:
            low = data[t-i]
    return (data[t] - abs(low)) / (high - abs(low)) # returns value from range [0, 1]


def vol_osc(t, volume, period1, period2):
    """
    Volume oscilator is calculated using two simple moving averages
    Args:
        t: current time step
        data: data array
        period1: the period of short term simple moving average
        period2: the period of long term simple moving average
    Returns:
        volume oscilator
    """
    short_sma = sma(t, volume, period1)
    long_sma = sma(t, volume, period2)
    return (short_sma - long_sma) / short_sma # returns value from range [0, 1]