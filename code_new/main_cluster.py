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
import wandb
import pandas as pd


def gather_data(stocks):
    """ Extracts data from txt files. We only use past returns and volume data
    :arg
        stocks: list of stock symbols (tickers)
    :returns
        tuple of returns and volumes to market cap
    """
    s = []
    stocks_returns = [f'data/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers))
        # print(f"Processing {stck_rtrn}, with {len(numbers)} stock returns")
    s = np.column_stack(s)
    
    v = []
    stocks_volumes = [f'data/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    for stck_vol in stocks_volumes:
        numbers = []
        with open(stck_vol, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        v.append(np.array(numbers))
        # print(f"Processing {stck_vol}, with {len(numbers)} stock volumes")
    v = np.column_stack(v)
    
    return s, v



def model_run(t_0, terminal_time, y, x, V_prior, G_prior, delta_0, alpha_0, beta_0, gamma_0, N, rho, tckrs):
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

    sign_accuracy = np.zeros(N)
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

    D = 0.005*np.eye(N)
    Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [1]
    rewards_agg = [1]
    rewards_even = [1]

    # omega_arr = []
    # forget_params = []

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
        for j in range(N):
            if np.sign(y_hat[j]) == np.sign(y[t][j]):
                sign_accuracy[j] += 1

        # mse += np.dot(e_t, e_t)
        # 2) Optimize allocation

        sigma = 1 / delta_t * Gyy @ Gyy.T
        omega = func.get_omega(D, sigma, actions[-1])
        # omega_arr.append(omega)
        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        A = act.update_evolution_matrix(N, rho, P, X, A)
        s_t = np.hstack((actions[-1], np.zeros(N), y_hat, x[t]))

        a_t = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 10,  P, Gxx, np.linalg.inv(sigma), sampling=False)

        a_t_greedy = act.action_generation(N, rho, s_t, B, Q_sqrt_greedy, A, 10,  P, Gxx, np.linalg.inv(sigma), sampling=False)

        actions_agg.append(a_t_greedy)
        actions.append(a_t)
        actions_even.append(a_even)
        # print("action for: ", y[t], "is: ", a_t, "estimated y: ", y_hat)
        tr_c = 0
        tr_c_agg = 0
        for i in range(N):
            tr_c += D[i][i] * np.abs(actions[-1][i] - a_t[i])
            tr_c_agg += D[i][i] * np.abs(actions_agg[-1][i] - a_t_greedy[i])
        reward = (1 + a_t @ y[t]) * (1 - tr_c)
        reward_agg = (1 + a_t_greedy @ y[t]) * (1 - tr_c_agg)
        reward_even = (1 + a_even @ y[t])
        rewards.append(rewards[-1] * reward)
        rewards_agg.append(rewards_agg[-1] * reward_agg)
        rewards_even.append(rewards_even[-1] * reward_even)

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
        # forget_params.append((alpha, beta, gamma))
        # print("alpha: ", alpha, "beta: ", beta, "gamma: ", gamma)

        G = func.update_G(G, d, G_0, alpha, beta)
        delta_t = (alpha + beta) * delta_t + beta + gamma * delta_0

        # V_naive = (alpha + beta) * V_naive + beta * np.outer(d, d) + gamma * V_prior
        # V_simple += np.outer(d, d)


    m_dd, var_r, avg_r = eval.risk_metrics(rewards)
    m_dd_agg, var_r_agg, avg_r_agg = eval.risk_metrics(rewards_agg)
    m_dd_even, var_r_even, avg_r_even = eval.risk_metrics(rewards_even)
    print(rewards[-1], rewards_even[-1])

    return np.array([m_dd, var_r, avg_r, m_dd_agg, var_r_agg, avg_r_agg, m_dd_even, var_r_even, avg_r_even])


if __name__ == "__main__":
    # 0) Select optional parameters

    ##  ==== params in prior ====
    mu = 0.8 # [0.1, 0.9) # set higher for sampling
    ##  forgetting params
    alpha0 = 0.1 # [0.02, 0.3]
    beta0 = 0.6 # [0.2, 0.6]
    gamma0 = 1 - alpha0 - beta0 # [0.2, 0.4]

    # ==== GA params ====
    batch = 4 # number of mutations in each iteration of the population
    p_m = 0.5 # mutation probability
    m_iter = 5*1e2 # maximum number of iterations of the GA
    decay = 0.992 # decay rate for p_m

    # 1) Collect arrays y and x
    all_tickers = ["AAON", "AAPL", "AIT", "AMD", "AMGN", "AMZN", "AVGO", "AXP", "BA", "CAT", "COST",
                   "CRM", "CSCO", "CVX", "DIS", "FIX", "GOOGL", "GS", "HD", "HON", "IBM", "JNJ",
                   "JPM", "KO", "META", "MMM", "MSFT", "MSTR", "NFLX", "NVDA", "PEP", "PFE", "PG",
                   "SMCI", "VZ", "WMT", "XOM"] # 40 stocks total
    results = pd.DataFrame(columns=["Portfolio", "Max Draw-Down: Risk-Averse", "Returns Variance: Risk-Averse",
                                    "Average Return: Risk-Averse", "Max-Draw-Down: Risk-Seeking",
                                    "Returns Variance: Risk-Seeking",
                                    "Average Return: Risk-Seeking", "Max Draw-Down: Uniform",
                                    "Returns Variance: Uniform",
                                    "Average Return: Uniform", "SE Parent"])
    n_tries = 20
    for i in range(n_tries):
        print("Try", i)
        n_assets = np.random.randint(low=2, high=4)
        t_bar = 10*n_assets
        T = int(t_bar*1.2)
        tickers = np.random.choice(all_tickers, n_assets)

        ret_full, vol_full = gather_data(tickers)

        ret = ret_full[len(ret_full)-256-T:]
        vol = vol_full[len(ret_full)-256-T:]

        # The regressor at time t: X[t] has to be constructed such it predicts y[t]
        # We construct it before the simulation run to keep everything tidy and easy to check

        p = 10
        # X = indicators.build_reduced_regressor(ret, vol, p) # after structure estimation
        X = indicators.build_regressor(ret, vol, p) # before structure estimation

        # set y = ret[p+1:], where p is the lag in AR(p), then X[t] is regressor for y[t], P@X[t] = y_hat for y[t]
        y = ret[p+1:]

        N = y.shape[1]
        # print(N)
        rho = X.shape[1]
        # print(X.shape)

        # 2) Collect \bar{V} and construct (V_0, delta_0), select priors alpha_0, beta_0, gamma_0

        V0, G0, delta0 = func.opt_prior(y, X, t_bar, N, rho, mu)
        delta0 += rho
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
        # Lambda = Vyy - Vyx @ np.linalg.inv(Vxx) @ Vyx.T
        r, l_max = func.genetic_algorithm(Vyy, Vyy0, Vyx, Vyx0, Vxx, Vxx0, T-t_bar, delta0, mu, batch, p_m, m_iter, decay)
        # print(f"best parent:{r}, with likelihood:{l_max}")
        r = np.asarray(r)
        # print(r.shape)
        X_se = X[:, r.astype(bool)]
        G0yy_se = G0[:N, :N]
        G0xx_se = func.reduce_matrix(G0[N:, N:], r.astype(bool), r.astype(bool))
        G0yx_se = func.reduce_matrix(G0[:N, N:], None, r.astype(bool))
        G0_se = np.block([[G0yy_se, G0yx_se], [np.zeros_like(G0yx_se.T), G0xx_se]])
        print(np.sum(r))
        # print(X_se.shape)
        # 4) Run the model

        T_terminal = len(y)
        try:
            res = model_run(T, T_terminal, y, X_se, V0, G0_se, delta0, alpha0, beta0, gamma0, N, np.sum(r), tickers)
            results.loc[len(results)] = [tickers] + res.tolist() + [r]
        except Exception as e:
            print(f"Run {i} failed: {e}")
            continue

    results.to_csv("results/res_small_portfolios.csv", index=False)

    print("FINISHED")