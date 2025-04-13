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
            if i-1 >= 0:
                transaction_costs += costs * np.abs(actions[i][j] - actions[i-1][j])
            else:
                actions[i-1][j] = 1/len(actions[0])
                transaction_costs += costs * np.abs(actions[i][j] - actions[i-1][j])
        utility += np.log(1 + data[i].T @ actions[i]) + np.log(1 - transaction_costs)
        print(f"previous action: {actions[i-1]}")
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
    plt.figure(figsize=(8, 4))
    plt.plot(time, rewards, label='Cumulative Gain', color='black', marker='o', markersize=4)
    plt.title('Cumulative Returns Over Time', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Gain')
    plt.legend()
    plt.tight_layout()
    plt.show()
    


def residual_plots(predictions, residuals, tickers):
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
        plt.xlabel(f'Predicted values {tickers[i]}')
        plt.ylabel(f'Residuals {tickers[i]}')
        plt.title(f'Residuals vs Predicted {tickers[i]}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


def qq_plots(residuals, tickers):
    """QQ plots

    Args:
        residuals: array of residuals
    """
    num_vars = residuals.shape[1]

    for i in range(num_vars):
        fig, ax = plt.subplots(figsize=(6, 6))
        stats.probplot(residuals[:, i], dist="norm", plot=ax)

        # Style line and points
        line = ax.get_lines()[1]
        points = ax.get_lines()[0]
        line.set_color("#555555")
        line.set_linestyle("--")
        points.set_markerfacecolor("#333333")
        points.set_alpha(0.7)

        ax.set_title(f'Q-Q Plot: {tickers[i]}', fontsize=14)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.grid(True, linestyle="--", linewidth=0.5)

        plt.tight_layout()
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
        

def plot_multiple_rewards(actions_list, data, costs, labels):
    """
    Plot cumulative returns for multiple action strategies.
    
    Args:
        actions_list: list of action arrays
        data: data array
        costs: transaction costs
        labels: optional list of labels for each strategy
    """
    plt.figure(figsize=(8, 4))

    for i, actions in enumerate(actions_list):
        rewards = reward(actions, data, costs)
        label = labels[i]
        plt.plot(rewards, label=label, marker='o', markersize=2)

    plt.title('Cumulative Returns', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Gain')
    plt.legend()
    plt.tight_layout()
    plt.show()
