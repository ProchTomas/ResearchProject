import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def reward(actions, data, costs):
    """ Custom reward function
    Args:
        actions: actions array
        data: data array
        costs: transaction costs

    Returns:
        tuple: array of progressing reward gained
    """
    utility = 0
    r = []
    for i in range(len(data)):
        transaction_costs = 0
        for j in range(len(actions[0])):
            transaction_costs += costs * np.abs(actions[i][j] - actions[i-1][j])
        utility += np.log(1 + data[i].T @ actions[i]) + np.log(1 - transaction_costs)
        r.append(np.exp(utility))
    return np.array(r)


def plot_reward(actions, data, costs):
    """
    Plots the reward gained over time
    Arguments for the function reward
    Args:
        actions: actions array
        data: data array
        costs: transaction costs
    """
    
    rewards = reward(actions, data, costs)
    time = np.arange(len(rewards))
    plt.plot(time, rewards, label=f'cumulative gain', marker='o')
    plt.title('Returns')
    plt.xlabel('Time')
    plt.ylabel('returns')
    plt.legend()
    plt.show()
    


def residual_plots(predictions, residuals):
    """Create residual plots
    residuals against predicted values

    Args:
        predictions: array of predictions for the assets
        residuals: array of residuals for the assets
    """
    num_vars = residuals.shape[1]
    
    for i in range(num_vars):
        plt.figure()
        plt.scatter(predictions[:, i], residuals[:, i])
        plt.xlabel(f'Predicted values (Dimension {i + 1})')
        plt.ylabel(f'Residuals (Dimension {i + 1})')
        plt.title(f'Residuals vs Predicted (Dimension {i + 1})')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


def qq_plots(residuals):
    """QQ plots

    Args:
        residuals: array of residuals
    """
    num_vars = residuals.shape[1]

    for i in range(num_vars):
        plt.figure()
        stats.probplot(residuals[:, i], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot (Dimension {i + 1})')
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.show()
    

def residuals_time_plots(residuals):
    """Plot residuals over time, used to identify overall model performance over time 
    and to identify time dependencies

    Args:
        residuals: array of residuals
    """
    residual_norm = np.linalg.norm(residuals, axis=1)
    
    time = np.arange(len(residuals))

    # Plot the norm of residuals over time
    plt.figure(figsize=(8, 4))
    plt.plot(time, residual_norm, marker='o', linestyle='-')
    plt.axhline(1e-2, color='orange', linestyle='--', linewidth=1)
    plt.title('Norm of Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual Norm')
    plt.show()


def average_allocation_chart(actions, tickers, window):
    """Plots average allocation for the past n trading days
    Args:
        actions: array of actions (allocation)
        tickers: array of ticker symbols
        window: past n trading days
    """
    n = len(actions)
    time = np.arange(len(actions))
    rolling_avg = np.array([actions[max(0, i - window):i].mean(axis=0) for i in range(1, n+1)])
    
    step = window
    sampled_time = time[window-1::step]
    sampled_allocations = rolling_avg[window-1::step]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(sampled_time/step))
    for i, ticker in enumerate(tickers):
        ax.bar(sampled_time/step, sampled_allocations[:, i], bottom=bottom, label=ticker)
        bottom += sampled_allocations[:, i]

    # Labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel(f"Avg Allocation (Past {window} Days)")
    ax.set_title(f"Portfolio Allocation Over Time ({window}-Day Average)")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))

    plt.show()


# TODO add list of the particular assets, overall MSE
def performance_metrics(residuals):
    """metrics: Durbin-Watson, MSE, MAE, R2

    Args:
        residuals: array of residuals
    """
    num_vars = residuals.shape[1]
    n = len(residuals)
    
    for i in range(num_vars):
        mse = 0
        dw1 = 0
        dw2 = 0
        
        mse += np.dot(residuals[i])
        for j in range(n):
            if 0 < j:
                dw1 += np.power(residuals[j-1][i] - residuals[j][i], 2)
            dw2 += np.power(residuals[j][i], 2)
            
        mse /= n
        
        print(f"MSE for asset {i}: {mse}")
        print(f"D-W for assets {i}: {dw1 / dw2}")
        print("--------------------------------")


def max_drawdown(arr):
    """Calculates a very important metric: The maximum draw-down the strategy experienced
    If we only gain, it returns zero
    Args:
        arr: array of returns (rewards)
    Returns:
        Maximum draw-down
    """
    max_drawdown = 0
    current_high = arr[0]
    current_low = arr[0]

    for i in range(len(arr)):
        if arr[i] > current_high:
            current_high = arr[i]
            current_low = arr[i]
        if arr[i] < current_low:
            current_low = arr[i]
        if 1 - current_low / current_high > max_drawdown:
            max_drawdown = 1 - current_low / current_high

    return max_drawdown    
        
        