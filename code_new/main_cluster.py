import math
import numpy as np
import pandas as pd
import act
import func
import indicators
import eval
from numpy.linalg import det as det


def gather_data(stocks):
    """ Extracts data from txt files. We only use past returns and volume data """
    s = []
    stocks_returns = [f'data_spx/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers))
    return np.column_stack(s)


def get_current_regressor(y_hist, p, step, max_lookback=40):
    """
    Slices history strictly up to t-1 and builds the regressor.
    """
    if step <= p:
        dummy_window = np.zeros((max_lookback, y_hist.shape[1]))
        return indicators.build_step_regressor(dummy_window, p, step)

    start_idx = max(0, step - max_lookback)
    window = y_hist[start_idx:step]
    return indicators.build_step_regressor(window, p, step)


def get_expanding_sharpe(rewards):
    """Calculates the expanding Sharpe Ratio iteratively for the dataframe."""
    rewards = np.array(rewards)
    returns = np.diff(rewards) / rewards[:-1]

    srs = [0.0]  # t=0 has no SR
    for i in range(1, len(returns)):
        window = returns[:i]
        var = np.var(window, ddof=1) if len(window) > 1 else 1e-8
        sr = np.mean(window) / np.sqrt(max(var, 1e-8))
        srs.append(sr)

    # Pad to match rewards length
    srs.append(srs[-1])
    return srs


def save_run_to_df(run_id, tckr, rewards_dict):
    """Saves the cleanly formatted rewards and Sharpe Ratios to CSV."""
    data = {}

    # Compute expanding SR for every model we tracked
    for name, rewards in rewards_dict.items():
        data[f"reward_{name}"] = rewards
        data[f"sr_{name}"] = get_expanding_sharpe(rewards)

    max_len = max(len(v) for v in data.values())
    for key in data:
        arr = data[key]
        if len(arr) < max_len:
            data[key] = list(arr) + [None] * (max_len - len(arr))

    df = pd.DataFrame(data)
    df["tickers"] = [",".join(tckr)] + [None] * (max_len - 1)

    filename = f"results/run_{run_id}_results.csv"
    df.to_csv(filename, index=False)


def run_conditional_markowitz(y_hist, t_0, terminal_time, p, ga_mask, lookback=40, risk_aversion=2.0):
    """
    Conditional Markowitz using strictly t-1 dynamically generated regressors.
    """
    N = y_hist.shape[1]
    rewards = [1.0]
    w_prev = np.ones(N) / N

    for t in range(t_0, terminal_time):
        start_idx = max(0, t - lookback)
        y_window = y_hist[start_idx:t]

        if len(y_window) < 2:
            w_opt = w_prev
        else:
            # Reconstruct the strictly historical X window to run OLS
            x_window = np.array([get_current_regressor(y_hist, p, step)[ga_mask] for step in range(start_idx, t)])
            x_t = get_current_regressor(y_hist, p, t)[ga_mask]

            # Estimate Conditional Returns via OLS (pseudo-inverse for stability)
            beta_hat = np.linalg.pinv(x_window) @ y_window
            mu = x_t @ beta_hat

            # Estimate Covariance from residuals
            residuals = y_window - (x_window @ beta_hat)
            cov = np.cov(residuals, rowvar=False)
            cov = np.atleast_2d(cov) + np.eye(N) * 1e-8

            w_opt = act.markowitz_allocation(mu, cov, risk_aversion)

        realized_return = w_opt @ y_hist[t]
        rewards.append(rewards[-1] * (1 + realized_return))
        w_prev = w_opt

    return rewards


def run_simulation(t_0, terminal_time, y, p, G_prior, delta_0, priors, N, ga_mask):
    """Main state-space simulation loop."""
    phi_0 = priors['phi_0']
    rho = np.sum(ga_mask)
    D = 0.002 * np.eye(N)

    models = [
        {'type': 'optimal', 'G': G_prior.copy(), 'delta': delta_0},
        {'type': 'stress', 'G': G_prior.copy(), 'delta': delta_0}
    ]

    A, B, X_mat = act.initialize_evolution_matrices(N, rho)
    H, H_greedy = 1e-5 * np.eye(3 * N + rho), 1e-5 * np.eye(3 * N + rho)
    a_prev, a_prev_greedy, a_even = np.ones(N) / N, np.ones(N) / N, np.ones(N) / N

    rewards = [1.0]
    rewards_greedy = [1.0]
    rewards_even = [1.0]

    for t in range(t_0, terminal_time):
        # 1) Construct regressor dynamically and apply GA mask
        x_t = get_current_regressor(y, p, t)[ga_mask]

        # 2) Predict & Mix
        y_hats, v_hats, Ps, Gxxs = [], [], [], []
        for mod in models:
            G, delta = mod['G'], mod['delta']
            Gyy, Gyx, Gxx = G[:N, :N], G[:N, N:], G[N:, N:]
            P = Gyx @ np.linalg.inv(Gxx)
            y_hats.append(P @ x_t)
            v_hats.append(Gyy @ Gyy / (delta - 2) * (1 + x_t.T @ np.linalg.inv(Gxx @ Gxx.T) @ x_t))
            Ps.append(P)
            Gxxs.append(Gxx)

        opt_weights = func.opt_forecast_weights(np.array(y_hats), np.array(v_hats), np.ones(len(models)) / len(models))
        P_mix = sum(w * P for w, P in zip(opt_weights, Ps))
        Gxx_mix = sum(w * Gxx for w, Gxx in zip(opt_weights, Gxxs))
        y_hat_mix = P_mix @ x_t
        sigma = sum(
            w * (1 / mod['delta'] * mod['G'][:N, :N] @ mod['G'][:N, :N].T) for w, mod in zip(opt_weights, models))

        # 3) Allocate
        omega = func.get_omega(D, sigma, a_prev)
        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)

        A_new = act.update_evolution_matrix(N, rho, P_mix, X_mat, A)

        s_t = np.hstack((a_prev, np.zeros(N), y_hat_mix, x_t))
        s_t_greedy = np.hstack((a_prev_greedy, np.zeros(N), y_hat_mix, x_t))

        a_opt, H = act.action_generation(N, rho, s_t, B, Q_sqrt, A_new, 1, H, P_mix, Gxx_mix, np.linalg.inv(sigma),
                                         sampling=False)
        a_greedy, H_greedy = act.action_generation(N, rho, s_t_greedy, B, Q_sqrt_greedy, A_new, 1, H_greedy, P_mix,
                                                   Gxx_mix, np.linalg.inv(sigma), sampling=False)

        # 4) Rewards
        tr_c = sum(D[i][i] * np.abs(a_prev[i] - a_opt[i]) for i in range(N))
        tr_c_greedy = sum(D[i][i] * np.abs(a_prev_greedy[i] - a_greedy[i]) for i in range(N))

        rewards.append(rewards[-1] * (1 + a_opt @ y[t]) * (1 - tr_c))
        rewards_greedy.append(rewards_greedy[-1] * (1 + a_greedy @ y[t]) * (1 - tr_c_greedy))
        rewards_even.append(rewards_even[-1] * (1 + a_even @ y[t]))

        a_prev, a_prev_greedy = a_opt, a_greedy
        A = A_new

        # 5) Update Posteriors
        d = np.concatenate((y[t], x_t))
        G1yy, G1xx = models[0]['G'][:N, :N], models[0]['G'][N:, N:]

        e_hat = y_hats[0] - y[t]
        Gxx_sq, Gyy_sq = G1xx @ G1xx.T, G1yy @ G1yy.T

        args = (priors['alpha_0'], priors['beta_0'], priors['gamma_0'], delta_0 + t, delta_0,
                G1xx, det(G1xx), G_prior[N:, N:], det(G_prior[N:, N:]),
                G1yy, det(G1yy), G_prior[:N, :N], det(G_prior[:N, :N]),
                Gxx_sq, Gxx_sq + np.outer(x_t, x_t), G_prior[N:, N:] @ G_prior[N:, N:].T,
                Gyy_sq, Gyy_sq + 1 / (1 + x_t.T @ np.linalg.inv(Gxx_sq) @ x_t) * np.outer(e_hat, e_hat),
                G_prior[:N, :N] @ G_prior[:N, :N].T, x_t.T @ np.linalg.inv(Gxx_mix @ Gxx_mix.T) @ x_t,
                e_hat.T @ np.linalg.inv(Gyy_sq) @ e_hat)

        alpha, beta, gamma = func.opt_forget_factors(args)
        eta = func.compute_eta(G1yy, G1xx, y[t], y_hat_mix, x_t, models[0]['delta'], rho)

        models[0]['G'] = func.update_G(models[0]['G'], d, G_prior, alpha, beta)
        models[0]['delta'] = (alpha + beta) * models[0]['delta'] + beta + gamma * delta_0
        models[1]['G'] = func.update_G(models[1]['G'], d, G_prior, 0.65, eta * 0.3)

    return {'ra': rewards, 'rs': rewards_greedy, 'uni': rewards_even}


if __name__ == "__main__":
    mu = 0.9
    priors = {'alpha_0': 0.09, 'beta_0': 0.5, 'gamma_0': 1 - 0.09 - 0.5, 'phi_0': 0.1}

    # GA params
    batch, p_m, m_iter, decay = 8, 0.5, 5 * 1e2, 0.992

    all_tickers = ['MRNA', 'COP', 'MOH', 'AXON', 'WST', 'EQT', 'AZO', 'DXCM', 'DELL', 'EXR', 'ENPH', 'BG', 'CRWD',
                   'EPAM', 'OXY', 'TGT', 'TSLA', 'CF', 'ANET', 'TSCO']

    n_tries = 200
    for i in range(n_tries):
        n_assets = np.random.randint(low=8, high=12)
        t_bar, T = 125, 250
        p = 10

        tickers = np.random.choice(all_tickers, n_assets, replace=False)
        ret_full = gather_data(tickers)

        # Ensure we have enough data
        if len(ret_full) < 500 + T:
            continue

        y = ret_full[len(ret_full) - 500 - T:]
        N = y.shape[1]

        # 1) Build Initial History for V0 and GA (Up to T)
        X_hist = np.array([get_current_regressor(y, p, t) for t in range(0, T)])
        rho_full = X_hist.shape[1]

        # Bootstrap Prior
        V0, G0, delta0 = func.opt_prior(y[:t_bar], X_hist[:t_bar], t_bar, N, rho_full, mu)

        # Build V from t_bar to T for GA
        V = V0.copy()
        for t in range(t_bar, T):
            d = np.concatenate((y[t], X_hist[t]))
            V += np.outer(d, d)

        Vyy, Vyx, Vxx = V[:N, :N], V[:N, N:], V[N:, N:]
        Vyy0, Vyx0, Vxx0 = V0[:N, :N], V0[:N, N:], V0[N:, N:]

        # 2) Run GA Structure Estimation
        r_mask, _ = func.genetic_algorithm(Vyy, Vyy0, Vyx, Vyx0, Vxx, Vxx0, T - t_bar, delta0, mu, batch, p_m, m_iter,
                                           decay)
        ga_mask = np.asarray(r_mask).astype(bool)

        # Fallback if GA strips all regressors
        if np.sum(ga_mask) == 0:
            ga_mask[0] = True

            # Reduce G0 based on GA selection
        G0yy_se = G0[:N, :N]
        G0xx_se = func.reduce_matrix(G0[N:, N:], ga_mask, ga_mask)
        G0yx_se = func.reduce_matrix(G0[:N, N:], None, ga_mask)
        G0_se = np.block([[G0yy_se, G0yx_se], [np.zeros_like(G0yx_se.T), G0xx_se]])

        # 3) Run Simulations
        T_terminal = len(y)
        try:
            # Run State-Space Models
            sim_rewards = run_simulation(T, T_terminal, y, p, G0_se, delta0, priors, N, ga_mask)

            # Run Conditional Markowitz Benchmark
            markowitz_rewards = run_conditional_markowitz(y, T, T_terminal, p, ga_mask, lookback=40, risk_aversion=2.0)
            sim_rewards['markowitz'] = markowitz_rewards

            # Calculate final Sharpe Ratios for clean console output
            sr_dict = {name: get_expanding_sharpe(r)[-1] for name, r in sim_rewards.items()}

            print(
                f"Run {i:03d} | Tickers: {N} | SR Averse: {sr_dict['ra']:5.2f} | SR Seeking: {sr_dict['rs']:5.2f} | SR Uni: {sr_dict['uni']:5.2f} | SR Mark: {sr_dict['markowitz']:5.2f}")

            # Save strictly what you care about to DF
            save_run_to_df(i, tickers, sim_rewards)

        except Exception as e:
            print(f"Run {i} failed: {e}")
            continue

    print("\n--- ALL CLUSTER RUNS FINISHED ---")