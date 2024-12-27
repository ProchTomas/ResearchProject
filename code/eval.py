import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def reward(actions, data, costs):
    """
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
    return r


def plot_reward(actions, data, costs):
    """
    Plots the reward gained over time
    Arguments for the function reward
    Args:
        actions: actions array
        data: data array
        costs: transaction costs
    """
    # TODO Plot gained reward against different strategies and benchmark
    rewards = reward(actions, data, costs)
    time = np.arange(rewards)
    
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
    fig, axes = plt.subplots(1, num_vars, figsize=(6 * num_vars, 4), sharey=True)
    
    for i in range(num_vars):
        ax = axes[i] if num_vars > 1 else axes  # Handle single subplot case
        ax.scatter(predictions[:, i], residuals[:, i], alpha=0.7)
        ax.axhline(0, color='orange', linestyle='--', linewidth=1)  # Add a reference line at 0
        ax.set_title(f'Dependent variable {i + 1} residual plot')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
    
    plt.tight_layout()
    plt.show()


def qq_plots(residuals):
    """QQ plots

    Args:
        residuals: array of residuals
    """
    num_vars = residuals.shape[1]

    # Create subplots for each dependent variable
    fig, axes = plt.subplots(1, num_vars, figsize=(6 * num_vars, 4))

    for i in range(num_vars):
        ax = axes[i] if num_vars > 1 else axes  # Handle single subplot case
        stats.probplot(residuals[:, i], dist="norm", plot=ax)
        ax.set_title(f'Dependent variable {i + 1} QQ plot')

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
    plt.axhline(0, color='orange', linestyle='--', linewidth=1)
    plt.title('Norm of Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual Norm')
    plt.show()
        