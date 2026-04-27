import math
import numpy as np
import matplotlib.pyplot as plt
import act # functions regarding action generation
import func # utility functions for the model
import indicators # indicator functions that go into regressor x
import eval
import time
from numpy.linalg import det as det
from scipy.linalg import cholesky, qr, solve_triangular
# import wandb
import json


def run_markowitz_baselines(y_full, n_portfolios, k_assets, t_0, terminal_time, lookback=40, risk_aversion=1.0):
    """
    Simulates Markowitz Mean-Variance optimization over n randomly sampled portfolios.

    :arg y_full: Array of historical returns for the entire asset universe, shape (T, N_total)
    :arg n_portfolios: Number of random portfolios to simulate
    :arg k_assets: Number of assets to randomly sample per portfolio
    :arg t_0: Starting index for the simulation
    :arg terminal_time: Ending index
    :arg lookback: Window size to estimate expected returns and covariance
    :returns: List of cumulative reward arrays for each simulated portfolio
    """
    N_total = y_full.shape[1]
    all_portfolio_paths = []

    for i in range(n_portfolios):
        # 1) Sample k random assets from the universe
        sampled_idx = np.random.choice(N_total, k_assets, replace=False)

        rewards = [1.0]
        w_prev = np.ones(k_assets) / k_assets

        for t in range(t_0, terminal_time):
            # 2) Isolate the lookback window strictly up to t-1
            start_idx = max(0, t - lookback)
            window = y_full[start_idx:t, sampled_idx]

            if len(window) < 2:
                w_opt = w_prev  # Fallback if there isn't enough history
            else:
                # Estimate mu and Sigma
                mu = np.mean(window, axis=0)
                cov = np.cov(window, rowvar=False)

                # Regularize covariance matrix slightly to prevent singularities
                cov += np.eye(k_assets) * 1e-8

                # Optimize
                w_opt = act.markowitz_allocation(mu, cov, risk_aversion)

            # 3) Step forward and capture realized returns
            realized_return = w_opt @ y_full[t, sampled_idx]

            # (Optional) Add transaction costs here if you want an apples-to-apples comparison
            tr_c = np.sum(0.002 * np.abs(w_opt - w_prev))
            # tr_c = 0.0

            rewards.append(rewards[-1] * (1 + realized_return) * (1 - tr_c))
            w_prev = w_opt

        all_portfolio_paths.append(rewards)
        print(f"Random Portfolio {i + 1}/{n_portfolios} simulated.")

    return all_portfolio_paths


def run_conditional_markowitz_baselines(y_full, n_portfolios, k_assets, t_0, terminal_time, lookback=40, risk_aversion=1.0,
                            X_full=None):
    """
    Simulates Markowitz Mean-Variance optimization.
    If X_full is provided, runs Conditional Markowitz using X[t] to predict expected returns.
    Otherwise, defaults to Historical Average Markowitz.
    """
    N_total = y_full.shape[1]
    all_portfolio_paths = []

    for i in range(n_portfolios):
        sampled_idx = np.random.choice(N_total, k_assets, replace=False)
        rewards = [1.0]
        w_prev = np.ones(k_assets) / k_assets

        for t in range(t_0, terminal_time):
            start_idx = max(0, t - lookback)
            y_window = y_full[start_idx:t, sampled_idx]

            if len(y_window) < 2:
                w_opt = w_prev  # Fallback for burn-in
            else:
                if X_full is not None:
                    # --- CONDITIONAL MARKOWITZ (Using X_t) ---
                    X_window = X_full[start_idx:t]

                    # Estimate rolling beta using pseudo-inverse for numerical stability
                    beta_hat = np.linalg.pinv(X_window) @ y_window

                    # Forecast expected return for time t using X[t]
                    mu = X_full[t] @ beta_hat

                    # Covariance of the prediction errors (residuals)
                    residuals = y_window - (X_window @ beta_hat)
                    cov = np.cov(residuals, rowvar=False)
                else:
                    # --- TRADITIONAL MARKOWITZ (Historical Averages) ---
                    mu = np.mean(y_window, axis=0)
                    cov = np.cov(y_window, rowvar=False)

                # Ensure covariance is a 2D matrix and regularize to prevent singularities
                cov = np.atleast_2d(cov) + np.eye(k_assets) * 1e-8

                # Optimize
                w_opt = act.markowitz_allocation(mu, cov, risk_aversion)

            # Step forward and capture realized returns
            realized_return = w_opt @ y_full[t, sampled_idx]
            rewards.append(rewards[-1] * (1 + realized_return))
            w_prev = w_opt

        all_portfolio_paths.append(rewards)
        print(f"Random Portfolio {i + 1}/{n_portfolios} simulated.")

    return all_portfolio_paths


def clustered_volatility(tau, n, low=-0.01, high=0.01, cluster_prob=0.01, cluster_len=5, seed=None):
    """Generates an array with clustered volatility."""
    vol_arr = []
    rng = np.random.default_rng(seed)
    for _ in range(n):
        arr = np.zeros(tau)
        t = 0
        while t < tau:
            if rng.random() < cluster_prob:
                length = min(rng.integers(cluster_len // 2, cluster_len * 2), tau - t)
                arr[t:t + length] = rng.normal(0, (high - low), size=length)
                t += length
            else:
                arr[t] = rng.normal(0, 0.1)
                t += 1
        arr = np.clip(arr, low, high)
        vol_arr.append(arr)
    return np.array([list(x) for x in zip(*vol_arr)])


def gather_data(stocks):
    """ Extracts data from txt files. We only use past returns and volume data
    :arg
        stocks: list of stock symbols (tickers)
    :returns
        tuple of returns and volumes to market cap
    """
    s = []
    stocks_returns = [f'data_spx/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers))
        # print(f"Processing {stck_rtrn}, with {len(numbers)} stock returns")
    s = np.column_stack(s)
    
    # v = []
    # stocks_volumes = [f'data_spx/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    # for stck_vol in stocks_volumes:
    #     numbers = []
    #     with open(stck_vol, 'r') as file:
    #         for line in file:
    #             numbers.append(float(line.strip()))
    #     v.append(np.array(numbers))
    #     # print(f"Processing {stck_vol}, with {len(numbers)} stock volumes")
    # v = np.column_stack(v)
    
    return s


def get_current_regressor(y_hist, p, step, max_lookback=40, is_toy=False, X_toy=None):
    """
    Slices history strictly up to t-1 for real data,
    or returns the perfectly aligned X[t] for the toy example.
    """
    if is_toy:
        # In the toy example, X[t] is already constructed to predict y[t]
        return X_toy[step]

    if step <= p:
        # Burn-in period for real data
        dummy_window = np.zeros((max_lookback, y_hist.shape[1]))
        return indicators.build_step_regressor(dummy_window, p, step)

    # Real data: Strict slice up to t-1
    start_idx = max(0, step - max_lookback)
    window = y_hist[start_idx:step]

    return indicators.build_step_regressor(window, p, step)


def build_forget_args(alpha_0, beta_0, gamma_0, delta_0, t, G, G_0, y_t, y_hat, x_t, Gxx_mix):
    """Helper to assemble the massive argument tuple for optimal forgetting."""
    N = len(y_t)
    Gyy, Gxx = G[:N, :N], G[N:, N:]
    Gyy_0, Gxx_0 = G_0[:N, :N], G_0[N:, N:]
    e_hat = y_hat - y_t

    Gxx_sq = Gxx @ Gxx.T
    Gyy_sq = Gyy @ Gyy.T
    Gxx_inv = np.linalg.inv(Gxx_sq)

    return (
        alpha_0, beta_0, gamma_0, delta_0 + t, delta_0,
        Gxx, det(Gxx), Gxx_0, det(Gxx_0),
        Gyy, det(Gyy), Gyy_0, det(Gyy_0),
        Gxx_sq, Gxx_sq + np.outer(x_t, x_t), Gxx_0 @ Gxx_0.T,
        Gyy_sq, Gyy_sq + 1 / (1 + x_t.T @ Gxx_inv @ x_t) * np.outer(e_hat, e_hat),
        Gyy_0 @ Gyy_0.T, x_t.T @ np.linalg.inv(Gxx_mix @ Gxx_mix.T) @ x_t,
        e_hat.T @ np.linalg.inv(Gyy_sq) @ e_hat
    )


def step_predict(models, x_t, N):
    """Generates mixed predictions from multiple models."""
    y_hats, v_hats, Ps, Gxxs = [], [], [], []

    for mod in models:
        G, delta = mod['G'], mod['delta']
        Gyy, Gyx, Gxx = G[:N, :N], G[:N, N:], G[N:, N:]

        P = Gyx @ np.linalg.inv(Gxx)
        y_hat = P @ x_t
        v_hat = Gyy @ Gyy / (delta - 2) * (1 + x_t.T @ np.linalg.inv(Gxx @ Gxx.T) @ x_t)

        y_hats.append(y_hat)
        v_hats.append(v_hat)
        Ps.append(P)
        Gxxs.append(Gxx)

    if len(models) > 1:
        opt_weights = func.opt_forecast_weights(np.array(y_hats), np.array(v_hats), np.ones(len(models)) / len(models))

        # Aggregate
        P_mix = sum(w * P for w, P in zip(opt_weights, Ps))
        Gxx_mix = sum(w * Gxx for w, Gxx in zip(opt_weights, Gxxs))
        y_hat_mix = P_mix @ x_t

        sigma = sum(w * (1 / mod['delta'] * mod['G'][:N, :N] @ mod['G'][:N, :N].T) for w, mod in zip(opt_weights, models))
        return y_hat_mix, P_mix, sigma, Gxx_mix, y_hats, opt_weights
    else:
        sigma = 1/mod['delta']*mod['G'][:N, :N] @ mod['G'][:N, :N].T

        return y_hat, P, sigma, Gxx, y_hats, 1.0


def step_allocate(N, rho, y_hat, x_t, sigma, P, Gxx_mix, a_prev, A, B, X, H_prev, D, h, thompson=False, greedy=False):
    """Finds optimal portfolio allocation."""
    omega = func.get_omega(D, sigma, a_prev)
    Q_sqrt = act.get_loss_matrix_greedy(N, rho, D, 0, 0) if greedy else act.get_loss_matrix(N, rho, D, sigma, omega)

    A_new = act.update_evolution_matrix(N, rho, P, X, A)
    s_t = np.hstack((a_prev, np.zeros(N), y_hat, x_t))

    a_opt, H_new = act.action_generation(N, rho, s_t, B, Q_sqrt, A_new, h, H_prev, P, Gxx_mix, np.linalg.inv(sigma),
                                         sampling=thompson)

    return a_opt, H_new, A_new, omega


def step_update(models, y_t, x_t, y_hat_mix, y_hats, Gxx_mix, G_0, priors, time_step):
    """Updates the state matrices and parameters for all models."""
    alpha_0, beta_0, gamma_0, delta_0 = priors
    N, rho = len(y_t), len(x_t)
    d = np.concatenate((y_t, x_t))

    # Calculate base optimal forgetting using the first model
    args = build_forget_args(alpha_0, beta_0, gamma_0, delta_0, time_step, models[0]['G'], G_0, y_t, y_hats[0], x_t,
                             Gxx_mix)
    alpha, beta, gamma = func.opt_forget_factors(args)

    # Calculate stress signal (eta)
    G1yy, G1xx = models[0]['G'][:N, :N], models[0]['G'][N:, N:]
    eta = func.compute_eta(G1yy, G1xx, y_t, y_hat_mix, x_t, models[0]['delta'], rho)
    beta2 = 0.95*eta

    # Update models dynamically based on their type
    for mod in models:
        if mod['type'] == 'optimal':
            mod['G'] = func.update_G(mod['G'], d, G_0, alpha, beta)
            mod['delta'] = (alpha + beta) * mod['delta'] + beta + gamma * delta_0
        elif mod['type'] == 'stress':
            mod['G'] = func.update_G(mod['G'], d, G_0, 0.96 - beta2, beta2)
            mod['delta'] = (0.96 + beta2)*mod['delta'] + beta2 + 0.01*delta_0 # keeps its internal logic or static decay depending on spec

    return models, (alpha, beta, gamma), beta2


def run_simulation(t_0, terminal_time, y, X, V_prior, G_prior, delta_0, priors, N, rho, opt_steps, p, thompson, iit, is_toy=True):
    """Main simulation loop handling state transitions over time."""
    phi_0 = priors['phi_0']
    D = 0.002 * np.eye(N)

    # Initialize multi-model setup
    models = [
        {'type': 'optimal', 'G': G_prior.copy(), 'delta': delta_0},
        # {'type': 'stress', 'G': G_prior.copy(), 'delta': delta_0}
    ]

    A, B, X_mat = act.initialize_evolution_matrices(N, rho)
    H, H_greedy = 1e-5 * np.eye(3 * N + rho), 1e-5 * np.eye(3 * N + rho)
    a_prev, a_prev_greedy, a_even = np.ones(N) / N, np.ones(N) / N, np.ones(N) / N

    metrics = {
        'rewards': [1], 'rewards_greedy': [1], 'rewards_even': [1],
        'actions': [a_prev], 'actions_greedy': [a_prev_greedy],
        'residuals': [], 'predictions': [], 'omega': [], 'forget_params': [], 'etas': [], 'phi': [], 'mix': [],
    }

    for t in range(t_0, terminal_time):
        if t % 100 == 0: print(f'Step: t={t}')

        # 1) Construct regressor dynamically (No Look-Ahead)
        x_t = get_current_regressor(y, p=p, step=t, max_lookback=40, is_toy=is_toy, X_toy=X)

        # 2) Predict & Mix
        y_hat_mix, P_mix, sigma, Gxx_mix, y_hats, weights = step_predict(models, x_t, N)
        metrics['residuals'].append(y[t] - y_hat_mix)
        metrics['predictions'].append(y_hat_mix)
        metrics['mix'].append(weights)

        # 3) Allocate
        a_opt, H_new, A, omega = step_allocate(N, rho, y_hat_mix, x_t, sigma, P_mix, Gxx_mix, a_prev, A, B, X_mat, H, D,
                                               opt_steps, thompson, greedy=False)
        a_greedy, H_new_greedy, _, _ = step_allocate(N, rho, y_hat_mix, x_t, sigma, P_mix, Gxx_mix, a_prev_greedy, A, B,
                                                     X_mat, H_greedy, D, opt_steps, thompson, greedy=True)

        # 4) Compute Rewards
        tr_c = sum(D[i][i] * np.abs(a_prev[i] - a_opt[i]) for i in range(N))
        tr_c_greedy = sum(D[i][i] * np.abs(a_prev_greedy[i] - a_greedy[i]) for i in range(N))

        metrics['rewards'].append(metrics['rewards'][-1] * (1 + a_opt @ y[t]) * (1 - tr_c))
        metrics['rewards_greedy'].append(metrics['rewards_greedy'][-1] * (1 + a_greedy @ y[t]) * (1 - tr_c_greedy))
        metrics['rewards_even'].append(metrics['rewards_even'][-1] * (1 + a_even @ y[t]))

        a_prev, a_prev_greedy = a_opt, a_greedy
        metrics['actions'].append(a_opt)
        metrics['actions_greedy'].append(a_greedy)
        metrics['omega'].append(omega)

        # 5) Update Parameters
        prior_tuple = (priors['alpha_0'], priors['beta_0'], priors['gamma_0'], delta_0)
        models, f_params, eta = step_update(models, y[t], x_t, y_hat_mix, y_hats, Gxx_mix, G_prior, prior_tuple, t)

        if iit:
            H, phi_opt = func.optimize_H(phi_0, H, H_new)
            H_greedy, _ = func.optimize_H(phi_0, H_greedy, H_new_greedy)
            metrics['phi'].append(phi_opt)

        metrics['forget_params'].append(f_params)
        metrics['etas'].append(eta)
    return metrics


if __name__ == "__main__":
    np.random.seed(0)

    sim_setup = {
        'T_bar': 10, 'T': 500,
        'N': 5, 'rho': 25,
        'mu': 0.9,
        'h_steps': 1, 'Thompson': False, 'iter_in_time': True, # when iter_in_time: True, set h_steps: 1
        'lag': 3
    }

    priors = {
        'alpha_0': 0.09, 'beta_0': 0.5, 'phi_0': 0.1,
        'gamma_0': 1 - 0.09 - 0.5
    }

    tickers = ['MRNA', 'TSLA', 'TGT', 'OXY', 'DELL']
    y_full = gather_data(tickers)
    y = y_full[:-1-sim_setup['T']-sim_setup['T_bar']]
    X = None

    # Toy Example Data Generation
    # beta_true = np.random.randn(sim_setup['rho'], sim_setup['N'])
    # X = np.random.randn(sim_setup['T'], sim_setup['rho']) * 1 / 100
    # noise = np.linspace(0.001, 0.005, sim_setup['T']).reshape(-1, 1) * np.random.randn(sim_setup['T'], sim_setup['N'])
    #
    # y = X @ beta_true + noise + clustered_volatility(sim_setup['T'], sim_setup['N'], seed=0)

    # Bootstrapping prior
    V0, G0, delta0 = func.opt_prior(y, sim_setup['T_bar'], sim_setup['N'], sim_setup['rho'], sim_setup['mu'], sim_setup['lag'])

    # Run modular simulation
    metrics = run_simulation(sim_setup['T_bar'], sim_setup['T'], y, X, V0, G0, delta0, priors, sim_setup['N'], sim_setup['rho'],
                             sim_setup['h_steps'], sim_setup['lag'], sim_setup['Thompson'], sim_setup['iter_in_time'], is_toy=False)

    # Delegate all evaluation and charting to eval.py
    # eval.evaluate_simulation_results(metrics, [str(i) for i in range(sim_setup['N'])])
    # print("FINISHED")


    # # --- 2. Run Markowitz Baselines ---
    print("\n--- Running Conditional Markowitz Baseline ---")
    markowitz_paths = run_markowitz_baselines(
        y_full=y,
        n_portfolios=1,  # Only need 1 run if k_assets == total universe N
        k_assets=sim_setup['N'],  # Assets per portfolio
        t_0=sim_setup['T_bar'],
        terminal_time=sim_setup['T'],
        lookback=40,
        risk_aversion=2.0,
        # X_full= # <--- PASS X HERE for Conditional Markowitz
    )

    # --- 3. Evaluate and Compare ---
    eval.evaluate_simulation_results(metrics, [str(i) for i in range(sim_setup['N'])], markowitz_paths)
    print("FINISHED")


