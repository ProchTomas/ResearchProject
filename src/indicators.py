import numpy as np


def ema_step(data, period):
    """Calculates the final EMA value for the given data window."""
    alpha = 2.0 / (period + 1.0)
    ema_val = data[0]
    for t in range(1, len(data)):
        ema_val = alpha * data[t] + (1.0 - alpha) * ema_val
    return ema_val


def rsi_step(data, period=14):
    """Calculates the custom RSI for the end of the window."""
    if len(data) < period + 1:
        return np.zeros(data.shape[1])

    diff = np.diff(data[-period - 1:], axis=0)
    gains = np.maximum(diff, 0).mean(axis=0)
    losses = np.maximum(-diff, 0).mean(axis=0)

    # Avoid division by zero
    rs = np.divide(gains, losses, out=np.zeros_like(gains), where=losses != 0)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # Custom transformation from your original code
    rsi_val = 1 / (1 + np.exp(0.4 * (rsi_val - 30))) + 1 / (1 + np.exp(0.4 * (rsi_val - 70))) - 1
    return rsi_val


def rolling_variance(data):
    """Returns the variance of the provided window."""
    if len(data) < 2: return np.zeros(data.shape[1])
    return np.var(data, axis=0, ddof=1)


def rolling_drawdown(data):
    """Calculates the max drawdown observed in the provided window."""
    if len(data) == 0: return np.zeros(data.shape[1])
    cum_returns = np.cumprod(1 + data, axis=0)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / np.maximum(running_max, 1e-8)
    return np.min(drawdowns, axis=0)


def build_step_regressor(window, p, step):
    """
    Builds the regressor x_t using strictly historical data up to t-1.
    :arg window: historical returns, shape (W, N)
    :arg p: AR lag
    :arg step: current time step t
    """
    W, N = window.shape

    # 1. Autoregressive Lags
    lags = window[-p:].flatten() if W >= p else np.zeros(p * N)

    # 2. Moving Averages / MACD
    if W >= 10:
        macd = ema_step(window, 10) - ema_step(window, 5)
    else:
        macd = np.zeros(N)

    # 3. Rolling Metrics (Example: 10-period window)
    if W >= 10:
        roll_var = rolling_variance(window[-10:])
        roll_dd = rolling_drawdown(window[-10:])
    else:
        roll_var, roll_dd = np.zeros(N), np.zeros(N)

    # 4. Oscillators
    rsi_val = rsi_step(window, period=14)

    # 5. Time features
    sin30 = np.sin(2 * np.pi * step / 30)

    # --- EASY TOGGLE BOARD ---
    # Comment or uncomment features below to add them to your regressor.
    features = [
        lags,
        macd,
        roll_var,
        # roll_dd,
        # rsi_val,
        # [sin30],
        # [1.0]  # Intercept
    ]

    return np.concatenate(features)