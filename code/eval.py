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
    # TODO Plot gained reward against different strategies and benchmark and color them
    
    rewards = reward(actions, data, costs)
    time = np.arange(len(rewards))
    print(time)
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
        plt.yscale('log') # log scale if some residual values are too big
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
    plt.axhline(0, color='orange', linestyle='--', linewidth=1)
    plt.title('Norm of Residuals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Residual Norm')
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
        mae = 0
        dw1 = 0
        dw2 = 0
        
        for j in range(n):
            mse += np.power(residuals[j][i], 2)
            mae += np.abs(residuals[j][i])
            if 0 < j:
                dw1 += np.power(residuals[j-1][i] - residuals[j][i], 2)
            dw2 += np.power(residuals[j][i], 2)
            
        mse /= n
        mae /= n
        
        print(f"MSE for asset {i}: {mse}")
        print(f"MAE for asset {i}: {mae}")
        print(f"D-W for assets {i}: {dw1 / dw2}")
        print("--------------------------------")
    
        