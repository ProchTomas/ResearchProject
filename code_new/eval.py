import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def residual_plots(predictions, residuals, tickers):
    """Create residual plots
    residuals against predicted values

    :arg
        predictions: array of predictions for the assets
        residuals: array of residuals for the assets
    """
    num_vars = residuals.shape[1]
    
    for i in range(num_vars):
        plt.figure()
        plt.scatter(predictions[:, i], residuals[:, i])
        plt.xlabel(f'fitted values {tickers[i]}')
        plt.ylabel(f'residuals {tickers[i]}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


def qq_plots(residuals, tickers):
    """QQ plots

    :arg
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

        ax.set_title(f'Normal Q-Q Plot: {tickers[i]}', fontsize=14)
        ax.set_xlabel("theoretical quantiles")
        ax.set_ylabel("sample quantiles")
        ax.grid(True, linestyle="--", linewidth=0.5)

        plt.tight_layout()
        plt.show()
    

def residuals_time_plots(residuals):
    """Plot residuals over time, used to identify overall model performance over time 
    and to identify time dependencies

    :arg
        residuals: array of residuals
    """
    residual_norm = np.linalg.norm(residuals, axis=1)
    
    time = np.arange(len(residuals))

    # Plot the norm of residuals over time
    plt.figure(figsize=(8, 4))
    plt.plot(time, residual_norm, marker='o', linestyle='-')
    plt.axhline(1e-2, color='orange', linestyle='--', linewidth=1)
    plt.xlabel('time')
    plt.ylabel('norm of residuals')
    plt.show()


def average_allocation_chart(actions, tickers, window):
    """Plots average allocation for the past n trading days
    :arg
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
    # Cold color palette (blue-purple gradient)
    cmap = plt.cm.Blues  # or try: plt.cm.Blues, plt.cm.twilight_shifted
    colors = cmap(np.linspace(0.2, 0.9, len(tickers)))  # skip extremes for nicer tone

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(sampled_time/step))
    for i, ticker in enumerate(tickers):
        ax.bar(sampled_time/step, sampled_allocations[:, i], bottom=bottom, label=ticker, color=colors[i])
        bottom += sampled_allocations[:, i]

    # Labels and legend
    ax.set_title(f"Portfolio Allocation ({window}-day average)")
    ax.set_xlabel("time")
    ax.set_ylabel(f"avg. allocation")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))

    plt.show()


def risk_metrics(cum_returns):
    """
    Calculate maximum draw-down and variance of returns

    :arg
        cum_returns : array of cumulative returns

    :returns
        max_drawdown : Maximum drawd-own in the cumulative returns
        var_returns : Variance of returns
    """
    cum_returns = np.array(cum_returns)

    # Compute simple returns
    simple_returns = np.diff(cum_returns) / cum_returns[:-1]

    # Variance of returns
    var_returns = np.var(simple_returns, ddof=1)
    avg_return = np.mean(simple_returns)

    # Maximum drawdown
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    return max_drawdown, var_returns, avg_return



def plot_perf(returns, ticker):
    """ Plots stocks performance from raw returns
    :arg
        returns: array of returns
    """
    cumulative_returns = [1]
    for t in range(len(returns)):
        cumulative_returns.append((cumulative_returns[-1] * (1 + returns[t])).astype(float))

    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    plt.plot(cumulative_returns, color="black")
    plt.title(f"{ticker}")
    plt.xlabel('time')
    plt.ylabel('cumulative returns')
    plt.show()


def plot_rewards(r):
    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    plt.plot(r, color="black")
    plt.xlabel('time')
    plt.ylabel('cumulative returns')
    plt.show()

def compare_rewards(rewards_list, labels_list):
    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    cmap = plt.cm.Greys_r  # or try: plt.cm.Blues, plt.cm.twilight_shifted
    colors = cmap(np.linspace(0.0, 0.6, len(rewards_list)))  # skip extremes for nicer tone
    for i, r in enumerate(rewards_list):
        plt.plot(r, color=colors[i], label=labels_list[i])
    plt.legend(frameon=False)
    plt.xlabel('time')
    plt.ylabel('cumulative returns')
    plt.show()


def plot_forgetting_params(params):
    colors = ["navy", "steelblue", "dodgerblue", "darkslategrey", "slategrey", "darkturquoise"]
    data = list(zip(*params))  # unzip into 3 sequences
    x = range(len(data[0]))  # index on x-axis
    labels = ("alpha", "beta", "gamma")
    fig, ax = plt.subplots()
    for series, label, color in zip(data, labels, colors[len(labels):]):
        ax.scatter(x, series, marker="o", label=label, color=color)

    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    # plt.title("Values over index")
    ax.grid(True)
    ax.legend()  # no box around legend
    plt.tight_layout()
    plt.show()

def plot_omega(o):
    x = range(len(o))
    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    plt.scatter(x, o, marker="o", label="omega", color="purple")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.show()

def plot_simulation_wealth_paths(r_arr):
    """Plots different wealth paths from the Thompson sampling simulation
    :arg
        r_arr : array of different cumulative returns
    """
    plt.gca().set_facecolor("white")
    plt.grid(True, which="both", linestyle=":", linewidth=0.7, color="lightgray", alpha=0.7)
    for i in range(len(r_arr)):
        plt.plot(r_arr[i], color="lightgrey")

    # Compute average and median across simulations (axis 0)
    # avg_path = np.mean(r_arr, axis=0)
    median_path = np.median(r_arr, axis=0)

    # Plot average and median paths
    # plt.plot(avg_path, color="darkblue", linewidth=2, label="average path")
    plt.plot(median_path, color="darkblue", linewidth=2, label="median path")

    plt.title("Sample Paths")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend(frameon=False)
    plt.show()


def evaluate_simulation_results(metrics, tickers):
    """
    Centralized function to print stats and plot graphs for the simulation output.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    m_dd, var_r, avg_r = risk_metrics(metrics['rewards'])
    m_dd_agg, var_r_agg, avg_r_agg = risk_metrics(metrics['rewards_greedy'])
    m_dd_even, var_r_even, avg_r_even = risk_metrics(metrics['rewards_even'])

    print('=== MODELS STATS ===')
    print(f"Risk-Averse  -> Max DD: {m_dd:.4f} | Var: {var_r:.6f} | Avg Return: {avg_r:.6f}")
    print(f"Risk-Seeking -> Max DD: {m_dd_agg:.4f} | Var: {var_r_agg:.6f} | Avg Return: {avg_r_agg:.6f}")
    print(f"Uniform      -> Max DD: {m_dd_even:.4f} | Var: {var_r_even:.6f} | Avg Return: {avg_r_even:.6f}")
    print("---------------------------")

    compare_rewards(
        [metrics['rewards'], metrics['rewards_greedy'], metrics['rewards_even']],
        ["risk-averse", "risk-seeking", "uniform"]
    )

    plot_omega(metrics['omega'])
    plot_forgetting_params(metrics['forget_params'])
    average_allocation_chart(np.array(metrics['actions']), tickers, 1)

    # Miscellaneous plots from main
    plt.figure()
    plt.plot(metrics['phi'], label="phi")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(metrics['etas'], label="eta")
    plt.xlabel("time")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(metrics['mix'], label='mixing weights')
    plt.xlabel("time")
    plt.legend()
    plt.show()