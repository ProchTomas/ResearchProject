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
import pandas as pd
import json


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
    # stocks_volumes = [f'data/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    # for stck_vol in stocks_volumes:
    #     numbers = []
    #     with open(stck_vol, 'r') as file:
    #         for line in file:
    #             numbers.append(float(line.strip()))
    #     v.append(np.array(numbers))
    #     # print(f"Processing {stck_vol}, with {len(numbers)} stock volumes")
    # v = np.column_stack(v)
    
    return s

import pandas as pd
import numpy as np

def save_run_to_df(run_id,
                   rewards, actions,
                   md_ra, rv_ra, ar_ra, sr_ra,
                   rewards_agg, actions_agg,
                   md_rs, rv_rs, ar_rs, sr_rs,
                   rewards_even, actions_even,
                   md_u, rv_u, ar_u, sr_u,
                   frgt_params, omegas,
                   tckr):

    # Put all arrays into a dict
    data = {
        "rewards": rewards,
        "actions": actions,
        "md_ra": md_ra,
        "rv_ra": rv_ra,
        "ar_ra": ar_ra,
        "sr_ra": sr_ra,
        "rewards_agg": rewards_agg,
        "actions_agg": actions_agg,
        "md_rs": md_rs,
        "rv_rs": rv_rs,
        "ar_rs": ar_rs,
        "sr_rs": sr_rs,
        "rewards_even": rewards_even,
        "actions_even": actions_even,
        "md_u": md_u,
        "rv_u": rv_u,
        "ar_u": ar_u,
        "sr_u": sr_u,
        "forget_params": frgt_params,
        "omegas": omegas,
    }

    # Find max length of all arrays so we can align the DataFrame
    max_len = max(len(v) for v in data.values())

    # Pad arrays so they have equal lengths (tuples stay untouched)
    for key in data:
        arr = data[key]
        if len(arr) < max_len:
            pad_len = max_len - len(arr)
            data[key] = list(arr) + [None] * pad_len

    # Create DataFrame
    df = pd.DataFrame(data)

    # Store portfolio tickers (as a single object)
    df["tickers"] = [tckr] + [None] * (max_len - 1)

    # Save file
    filename = f"results/run_{run_id}_results.csv"
    df.to_csv(filename, index=False)

    print(f"Saved run {run_id} → {filename}")


def model_run(run_id, t_0, terminal_time, y, x, V_prior, G_prior, delta_0, alpha_0, beta_0, gamma_0, N, rho, tckrs):
    """Runs the model on past data
    :arg
        t_0: initial time
        terminal_time: terminal time for the simulation
        y: array of response variables
        x: array of regressor values
        V_prior: prior statistics
        G_prior: prior square-root of the statistics
        delta_0: prior degrees of freedom
        alpha_0: prior forgetting parameters
        beta_0
        gamma_0
        N: dim(y)
    """
    G = G_prior.copy()
    G_0 = G_prior.copy()
    # V_naive = V_prior.copy()
    # V_simple = V_prior.copy()

    delta_t = delta_0
    Gyy_0 = G_0[:N, :N]
    Gyx_0 = G_0[:N, N:]
    Gxx_0 = G_0[N:, N:]

    # sign_accuracy = np.zeros(N)
    # mse = 0
    # residuals = []
    # predictions = []
    actions = []
    a_prev = np.ones(N)/N
    a_even = a_prev.copy()
    actions.append(a_prev)
    actions_agg = []
    actions_agg.append(a_prev)
    actions_even = []
    actions_even.append(a_prev)

    D = 0.002*np.eye(N)
    Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [1]
    rewards_agg = [1]
    rewards_even = [1]

    omega_arr = []
    forget_params = []
    md_ra = []
    md_rs = []
    md_u = []
    rv_ra = []
    rv_rs = []
    rv_u = []
    ar_ra = []
    ar_rs = []
    ar_u = []
    sr_ra = []
    sr_rs = []
    sr_u = []


    for t in range(t_0, terminal_time):
        if t%100 == 0:
            print(t)
        # 0) Partition G (upper triangular)
        Gyy = G[:N, :N]
        Gyx = G[:N, N:]
        Gxx = G[N:, N:]

        # 1) Get point estimates of P, Lambda

        P = Gyx @ np.linalg.inv(Gxx)
        y_hat = P@x[t]
        e_t = y[t] - y_hat
        # print("real observation:", y[t], "predicted observation:", y_hat, "used regressor:", x[t])
        # residuals.append(e_t)
        # predictions.append(y_hat)
        # for j in range(N):
        #     if np.sign(y_hat[j]) == np.sign(y[t][j]):
        #         sign_accuracy[j] += 1

        # mse += np.dot(e_t, e_t)
        # 2) Optimize allocation

        sigma = 1 / delta_t * Gyy @ Gyy.T
        omega = func.get_omega(D, sigma, actions[-1])
        omega_arr.append(omega)
        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        A = act.update_evolution_matrix(N, rho, P, X, A)
        s_t = np.hstack((actions[-1], np.zeros(N), y_hat, x[t]))
        s_t_greedy = np.hstack((actions_agg[-1], np.zeros(N), y_hat, x[t]))

        a_t = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 10,  P, Gxx, np.linalg.inv(sigma), sampling=False)

        a_t_greedy = act.action_generation(N, rho, s_t_greedy, B, Q_sqrt_greedy, A, 10,  P, Gxx, np.linalg.inv(sigma), sampling=True)

        actions_agg.append(a_t_greedy)
        actions.append(a_t)
        # print("action for: ", y[t], "is: ", a_t, "estimated y: ", y_hat)
        tr_c = 0
        tr_c_agg = 0
        for i in range(N):
            tr_c += D[i][i] * np.abs(actions[-1][i] - a_t[i])
            tr_c_agg += D[i][i] * np.abs(actions_agg[-1][i] - a_t_greedy[i])
        reward = (1 + a_t @ y[t]) * (1 - tr_c)
        reward_agg = (1 + a_t_greedy @ y[t]) * (1 - tr_c_agg)
        reward_even = (1 + actions_even[-1] @ y[t])
        rewards.append(rewards[-1] * reward)
        rewards_agg.append(rewards_agg[-1] * reward_agg)
        rewards_even.append(rewards_even[-1] * reward_even)

        c = np.dot(actions_even[-1], 1 + y[t])
        u_allocation = [a * (1 + b) / c for a, b in zip(actions_agg[-1], y[t])]
        actions_even.append(u_allocation)

        # 3) Update G

        d = np.concatenate((y[t], x[t]))

        # gather necessary arguments for function F(phi)
        args = (alpha_0, beta_0, gamma_0, delta_0 + t, delta_0, Gxx, det(Gxx), Gxx_0, det(Gxx_0), Gyy, det(Gyy), Gyy_0,
                det(Gyy_0), G @ G.T, G @ G.T + np.outer(d, d), G_0 @ G_0.T, Gyy @ Gyy.T,
                Gyy @ Gyy.T + 1/(1+x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t])*e_t @ e_t.T,
                Gyy_0 @ Gyy_0.T, x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t],
                e_t.T @ np.linalg.inv(Gyy @ Gyy.T) @ e_t)

        # Compute the optimal forgetting factors, gamma = 1 - alpha - beta
        alpha, beta, gamma = func.opt_forget_factors(args)
        forget_params.append((alpha, beta, gamma))

        G = func.update_G(G, d, G_0, alpha, beta)
        delta_t = (alpha + beta) * delta_t + beta + gamma * delta_0

        # V_naive = (alpha + beta) * V_naive + beta * np.outer(d, d) + gamma * V_prior
        # V_simple += np.outer(d, d)

        m_dd_ra, var_r_ra, avg_r_ra = eval.risk_metrics(rewards)
        md_ra.append(m_dd_ra)
        rv_ra.append(var_r_ra)
        ar_ra.append(avg_r_ra)
        sr_ra.append(avg_r_ra/np.sqrt(var_r_ra))

        m_dd_rs, var_r_rs, avg_r_rs = eval.risk_metrics(rewards_agg)
        md_rs.append(m_dd_rs)
        rv_rs.append(var_r_rs)
        ar_rs.append(avg_r_rs)
        sr_rs.append(avg_r_rs/np.sqrt(var_r_rs))

        m_dd_u, var_r_u, avg_r_u = eval.risk_metrics(rewards_even)
        md_u.append(m_dd_u)
        rv_u.append(var_r_u)
        ar_u.append(avg_r_u)
        sr_u.append(avg_r_u/np.sqrt(var_r_u))


    save_run_to_df(run_id, rewards, actions, md_ra, rv_ra, ar_ra, sr_ra, rewards_agg, actions_agg, md_rs, rv_rs, ar_rs, sr_rs,
                   rewards_even, actions_even, md_u, rv_u, ar_u, sr_u, forget_params, omega_arr, tckrs)


if __name__ == "__main__":
    # 0) Select optional parameters

    ##  ==== params in prior ====
    mu = 0.9 # first try: 0.9, second try: 0.5
    ##  forgetting params
    alpha0 = 0.09 # [0.02, 0.3]
    beta0 = 0.5 # [0.2, 0.6]
    gamma0 = 1 - alpha0 - beta0 # [0.2, 0.4]

    # ==== GA params ====
    batch = 8 # number of mutations in each iteration of the population
    p_m = 0.5 # mutation probability
    m_iter = 5*1e2 # maximum number of iterations of the GA
    decay = 0.992 # decay rate for p_m

    # 1) Collect arrays y and x
    # least correlated
    # all_tickers = ['MRNA', 'KR', 'TDG', 'SW', 'NEM', 'MOH', 'SMCI', 'CLX', 'EQT', 'FSLR', 'CBOE', 'K', 'NFLX', 'DLTR', 'WST', 'OXY', 'DPZ', 'SJM', 'GEN', 'LLY']
    # top 100 returns and 20 least correlated of them
    all_tickers = ['MRNA', 'COP', 'MOH', 'AXON', 'WST', 'EQT', 'AZO', 'DXCM', 'DELL', 'EXR', 'ENPH', 'BG', 'CRWD', 'EPAM', 'OXY',
     'TGT', 'TSLA', 'CF', 'ANET', 'TSCO']
    n_tries = 200 # was 200
    for i in range(n_tries):
        print("Try", i)
        n_assets = np.random.randint(low=8, high=12) # was set at 10
        t_bar = 125
        T = 250
        tickers = np.random.choice(all_tickers, n_assets)

        ret_full = gather_data(tickers)

        ret = ret_full[len(ret_full)-500-T:]
        # vol = vol_full[len(ret_full)-500-T:]

        # The regressor at time t: X[t] has to be constructed such it predicts y[t]
        # We construct it before the simulation run to keep everything tidy and easy to check

        p = 10
        # X = indicators.build_reduced_regressor(ret, vol, p) # after structure estimation
        X = indicators.build_regressor(ret, p) # before structure estimation

        # set y = ret[p+1:], where p is the lag in AR(p), then X[t] is regressor for y[t], P@X[t] = y_hat for y[t]
        y = ret[p+1:]

        N = y.shape[1]
        # print(N)
        rho = X.shape[1]
        # print(X.shape)

        # 2) Collect \bar{V} and construct (V_0, delta_0), select priors alpha_0, beta_0, gamma_0

        V0, G0, delta0 = func.opt_prior(y, X, t_bar, N, rho, mu)
        #
        Vyy0 = V0[:N, :N]
        Vyx0 = V0[:N, N:] # in text: Vyx = Vxy.T
        Vxx0 = V0[N:, N:]
        #
        # # # 3) Gather more data and run structure estimation
        # #
        V = V0.copy()
        for t in range(t_bar, T):
             d = np.concatenate((y[t], X[t]))
             V += np.outer(d, d)
        Vyy = V[:N, :N]
        Vyx = V[:N, N:]
        Vxx = V[N:, N:]
        # # Lambda = Vyy - Vyx @ np.linalg.inv(Vxx) @ Vyx.T
        r, l_max = func.genetic_algorithm(Vyy, Vyy0, Vyx, Vyx0, Vxx, Vxx0, T-t_bar, delta0, mu, batch, p_m, m_iter, decay)
        # # print(f"best parent:{r}, with likelihood:{l_max}")
        r = np.asarray(r)
        print(r)
        # # print(r.shape)
        X_se = X[:, r.astype(bool)]
        G0yy_se = G0[:N, :N]
        G0xx_se = func.reduce_matrix(G0[N:, N:], r.astype(bool), r.astype(bool))
        G0yx_se = func.reduce_matrix(G0[:N, N:], None, r.astype(bool))
        G0_se = np.block([[G0yy_se, G0yx_se], [np.zeros_like(G0yx_se.T), G0xx_se]])
        # print(np.sum(r))
        # print(X_se.shape)
        # 4) Run the model

        T_terminal = len(y)
        try:
            # model_run(i, T, T_terminal, y, X, V0, G0, delta0, alpha0, beta0, gamma0, N, rho, tickers)
            model_run(i, T, T_terminal, y, X_se, V0, G0_se, delta0, alpha0, beta0, gamma0, N, np.sum(r), tickers)
        except Exception as e:
            print(f"Run {i} failed: {e}")
            continue

    print("FINISHED")