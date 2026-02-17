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
import itertools
import csv


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
    s = np.column_stack(s)

    return s


def model_run(t_0, terminal_time, y, x, V_prior, G_prior, delta_0, alpha_0, beta_0, gamma_0, N, rho):
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
    V_naive = V_prior.copy()
    V_simple = V_prior.copy()

    delta_t = delta_0
    Gyy_0 = G_0[:N, :N]
    Gyx_0 = G_0[:N, N:]
    Gxx_0 = G_0[N:, N:]

    actions = []
    a_prev = np.ones(N)/N
    actions.append(a_prev)
    actions_agg = []
    actions_agg.append(a_prev)


    D = 0.002*np.eye(N)
    Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [1]
    rewards_agg = [1]


    for t in range(t_0, terminal_time):
        # 0) Partition G (upper triangular)
        if t%50 == 0:
            print(t)

        Gyy = G[:N, :N]
        Gyx = G[:N, N:]
        Gxx = G[N:, N:]

        # 1) Get point estimates of P, Lambda

        P = Gyx @ np.linalg.inv(Gxx)
        y_hat = P@x[t]
        e_t = y[t] - y_hat
        # print("real observation:", y[t], "predicted observation:", y_hat, "used regressor:", x[t])
        #residuals.append(e_t)
        #predictions.append(y_hat)
        # for j in range(N):
        #     if np.sign(y_hat[j]) == np.sign(y[t][j]):
        #         sign_accuracy[j] += 1
        #
        # mse += np.dot(e_t, e_t)
        # 2) Optimize allocation

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
        # print("alpha: ", alpha, "beta: ", beta, "gamma: ", gamma)

        G = func.update_G(G, d, G_0, alpha, beta)
        delta_t = (alpha + beta) * delta_t + beta + gamma * delta_0

        V_naive = (alpha + beta) * V_naive + beta * np.outer(d, d) + gamma * V_prior
        V_simple += np.outer(d, d)

        sigma = 1/delta_t * Gyy @ Gyy.T
        omega = func.get_omega(D, sigma, actions[-1])

        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        A = act.update_evolution_matrix(N, rho, P, X, A)
        s_t = np.hstack((actions[-1], np.zeros(N), y_hat, x[t]))
        s_t_greedy = np.hstack((actions_agg[-1], np.zeros(N), y_hat, x[t]))

        a_t = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 10,  P, Gxx, np.linalg.inv(sigma), sampling=True)
        a_t_greedy = act.action_generation(N, rho, s_t_greedy, B, Q_sqrt_greedy, A, 10, P, Gxx, np.linalg.inv(sigma), sampling=True)


        # print("action for: ", y[t], "is: ", a_t, "estimated y: ", y_hat)
        tr_c = 0
        tr_c_agg = 0
        for i in range(N):
            tr_c += D[i][i] * np.abs(actions[-1][i] - a_t[i])
            tr_c_agg += D[i][i] * np.abs(actions_agg[-1][i] - a_t_greedy[i])
        reward = (1 + a_t @ y[t]) * (1 - tr_c)
        reward_agg = (1 + a_t_greedy @ y[t]) * (1 - tr_c_agg)
        rewards.append(rewards[-1] * reward)
        rewards_agg.append(rewards_agg[-1] * reward_agg)

        actions.append(a_t)
        actions_agg.append(a_t_greedy)

    m_dd, var_r, avg_r = eval.risk_metrics(rewards)
    m_dd_agg, var_r_agg, avg_r_agg = eval.risk_metrics(rewards_agg)

    sharpe_ratio = avg_r/np.sqrt(var_r)
    sharpe_ratio_agg = avg_r_agg/np.sqrt(var_r_agg)

    res = {
        "reward ra": rewards[-1],
        "mdd ra:": m_dd,
        "sr ra": sharpe_ratio,
        "reward rs": rewards_agg[-1],
        "mdd rs": m_dd_agg,
        "sr rs": sharpe_ratio_agg,
    }

    return res


if __name__ == "__main__":
    t_bar = 30  # time taken to collect V_bar

    PARAM_GRID = {
            "alpha0": [0.05],
            "beta0": [0.7],
            "mu": [0.5],
            # "sampling": [True, False],
        }
    keys = PARAM_GRID.keys()
    design_points = list(itertools.product(*PARAM_GRID.values()))

    all_tickers = ['MRNA', 'COP', 'MOH', 'AXON', 'WST', 'EQT', 'AZO', 'DXCM', 'DELL', 'EXR', 'ENPH', 'BG', 'CRWD',
                   'EPAM', 'OXY',
                   'TGT', 'TSLA', 'CF', 'ANET', 'TSCO']

    rows = []
    for setting_id, values in enumerate(design_points):
        config_run = dict(zip(keys, values))

        alpha0 = config_run["alpha0"]
        beta0 = config_run["beta0"]
        gamma0 = 1 - alpha0 - beta0
        mu = config_run["mu"]
        # sampling = config_run["sampling"]

        for rep in range(5):  # run 5 replications
            tickers = np.random.choice(all_tickers, 5)
            ret_full = gather_data(tickers)

            ret = ret_full[:300]
            p = 4
            X = indicators.build_reduced_regressor(ret, p)
            y = ret[p + 1:]

            N = y.shape[1]
            rho = X.shape[1]
            T_terminal = len(y)
            V0, G0, delta0 = func.opt_prior(y, X, t_bar, N, rho, mu)
            try:
                result = model_run(t_bar, T_terminal,
                    y, X, V0, G0, delta0,
                    alpha0, beta0, gamma0,
                    N, rho)
            except Exception as e:
                result = {
                    "reward ra": 0,
                    "mdd ra:": 0,
                    "sr ra": 0,
                    "reward rs": 0,
                    "mdd rs": 0,
                    "sr rs": 0,
                }

            # assume result is dict-like; adapt if needed
            row = {
                "setting_id": setting_id,
                "replication": rep,
                **config_run,
                **result
            }

            rows.append(row)
    with open("results/model_results_center_point.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("FINISHED")