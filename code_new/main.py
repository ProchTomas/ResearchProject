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



def model_run(t_0, terminal_time, y, x, V_prior, G_prior, delta_0, alpha_0, beta_0, gamma_0, phi_0, N, rho, tckrs):
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
    V_simple = G_prior.copy()
    H = 1e-5*np.eye(3*N+rho)
    H_greedy = H.copy()

    delta_t = delta_0
    delta_simple = delta_0
    Gyy_0 = G_0[:N, :N]
    Gyx_0 = G_0[:N, N:]
    Gxx_0 = G_0[N:, N:]

    sign_accuracy = np.zeros(N)
    mse = 0
    residuals = []
    predictions = []
    actions = []
    a_prev = np.ones(N)/N
    a_even = a_prev.copy()
    actions.append(a_prev)
    actions_agg = []
    actions_agg.append(a_prev)
    actions_even = []
    actions_even.append(a_prev)

    D = 0.002*np.eye(N)
    # D = np.diag(np.array([0.002, 0.01]))
    Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [1]
    rewards_agg = [1]
    rewards_even = [1]

    omega_arr = []
    forget_params = []
    sr = []
    sr_a = []
    sr_u = []

    for t in range(t_0, terminal_time):
        # 0) Partition G (upper triangular)
        Gyy = G[:N, :N]
        Gyx = G[:N, N:]
        Gxx = G[N:, N:]

        # 1) Get point estimates of P, Lambda

        P = Gyx @ np.linalg.inv(Gxx)
        y_hat = P@x[t]
        e_t = y[t] - y_hat
        # print("real observation:", y[t], "predicted observation:", y_hat, "used regressor:", x[t])
        residuals.append(e_t)
        predictions.append(y_hat)
        for j in range(N):
            if np.sign(y_hat[j]) == np.sign(y[t][j]):
                sign_accuracy[j] += 1

        mse += np.dot(e_t, e_t)
        # 2) Optimize allocation

        sigma = 1 / delta_t * Gyy @ Gyy.T
        omega = func.get_omega(D, sigma, actions[-1])
        omega_arr.append(omega)
        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        A = act.update_evolution_matrix(N, rho, P, X, A)
        s_t = np.hstack((actions[-1], np.zeros(N), y_hat, x[t]))
        s_t_greedy = np.hstack((actions_agg[-1], np.zeros(N), y_hat, x[t]))

        a_t, H_new = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 10, H,  P, Gxx, np.linalg.inv(sigma), sampling=False)

        a_t_greedy, H_new_greedy = act.action_generation(N, rho, s_t_greedy, B, Q_sqrt_greedy, A, 10, H_greedy,  P, Gxx, np.linalg.inv(sigma), sampling=False)
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
        u_allocation = [a * (1+b) / c for a, b in zip(actions_agg[-1], y[t])]
        actions_even.append(u_allocation)

        # 3) Update G

        d = np.concatenate((y[t], x[t]))

        # gather necessary arguments for function F(phi)
        args = (alpha_0, beta_0, gamma_0, delta_0+t, delta_0, Gxx, det(Gxx), Gxx_0, det(Gxx_0), Gyy, det(Gyy), Gyy_0,
                det(Gyy_0), G @ G.T, G @ G.T + np.outer(d, d), G_0 @ G_0.T, Gyy @ Gyy.T,
                Gyy @ Gyy.T + 1/(1+x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t])*e_t @ e_t.T,
                Gyy_0 @ Gyy_0.T, x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t],
                e_t.T @ np.linalg.inv(Gyy @ Gyy.T) @ e_t)

        # Vyy = V_simple[:N, :N]
        # Vyx = V_simple[:N, N:]
        # Vxx = V_simple[N:, N:]

        # args = (alpha_0, beta_0, gamma_0, delta_0 + t, delta_0, Gxx, det(Gxx), Vxx, det(Vxx), Gyy, det(Gyy), Vyy,
        #         det(Vyy), G @ G.T, G @ G.T + np.outer(d, d), V_simple @ V_simple.T, Gyy @ Gyy.T,
        #         Gyy @ Gyy.T + 1 / (1 + x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t]) * e_t @ e_t.T,
        #         Vyy @ Vyy.T, x[t].T @ np.linalg.inv(Gxx @ Gxx.T) @ x[t],
        #         e_t.T @ np.linalg.inv(Gyy @ Gyy.T) @ e_t)

        # Compute the optimal forgetting factors, gamma = 1 - alpha - beta
        alpha, beta, gamma = func.opt_forget_factors(args)
        forget_params.append((alpha, beta, gamma))
        # if gamma > 0.9:
        #     V_simple = func.update_G(V_simple, d, G_0, 0, 0)

        # print("alpha: ", alpha, "beta: ", beta, "gamma: ", gamma)

        G = func.update_G(G, d, V_simple, alpha, beta)
        delta_t = (alpha+beta)* delta_t + beta + gamma * delta_0

        # H = func.optimize_H(phi_0, H, H_new)
        # H_greedy = func.optimize_H(phi_0, H, H_new_greedy)

        # V_naive = (alpha + beta) * V_naive + beta * np.outer(d, d) + gamma * V_prior
        md, vr, ar = eval.risk_metrics(rewards)
        md_a, vr_a, ar_a = eval.risk_metrics(rewards_agg)
        md_u, vr_u, ar_u = eval.risk_metrics(rewards_even)
        sr.append(ar/vr)
        sr_a.append(ar_a/vr_a)
        sr_u.append(ar_u/vr_u)



    m_dd, var_r, avg_r = eval.risk_metrics(rewards)
    m_dd_agg, var_r_agg, avg_r_agg = eval.risk_metrics(rewards_agg)
    m_dd_even, var_r_even, avg_r_even = eval.risk_metrics(rewards_even)
    print("max draw-down (risk-seeking): ", m_dd_agg)
    print("variance in portfolio returns (risk-seeking): ", var_r_agg)
    print("average reward (risk-seeking): ", avg_r_agg)
    print("---------------------------")
    print("max draw-down (uniform): ", m_dd_even)
    print("variance in portfolio returns (uniform): ", var_r_even)
    print("average reward (uniform): ", avg_r_even)
    print("---------------------------")
    eval.compare_rewards([rewards, rewards_agg, rewards_even], ["risk-averse", "risk-seeking", "uniform"])
    eval.plot_omega(omega_arr)
    eval.plot_forgetting_params(forget_params)
    eval.average_allocation_chart(np.array(actions), tckrs, 10)
    eval.average_allocation_chart(np.array(actions_agg), tckrs, 10)

    plt.plot(sr[10:], label="risk-averse")
    plt.plot(sr_a[10:], label="risk-seeking")
    plt.plot(sr_u[10:], label="uniform")
    plt.xlabel("time")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.show()

    return m_dd, var_r, avg_r, sign_accuracy


if __name__ == "__main__":
    # 0) Select optional parameters
    t_bar = 10  # time taken to collect V_bar
    # T = 250 # time taken to collect more data for structure estimation

    ##  ==== params in prior ====
    mu = 0.9 # [0.6, 0.9] # set higher for sampling
    ##  forgetting params
    alpha0 = 0.09 # [0.02, 0.3]
    beta0 = 0.5 # [0.2, 0.9]
    gamma0 = 1 - alpha0 - beta0 # [0.01, 0.4]
    phi0 = 0.1

    # ==== GA params ====
    batch = 16 # number of mutations in each iteration of the population
    p_m = 0.25 # mutation probability
    m_iter = 1*1e2 # maximum number of iterations of the GA
    decay = 0.99 # decay rate for p_m

    # 1) Collect arrays y and x
    tickers = ["1", "2", "3"]
    # tickers = ["AIT", "AAPL"] # test1 lag = 4
    # tickers = ["SPX", "NDX", "GLD"] # test2 lag = 4
    # tickers = ["AAON", "AIT", "MSTR", "SMCI", "FIX"] # test3 lag = 3
    # tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT"] # test4 lag = 4
    # tickers_full = ['MRNA', 'KR', 'TDG', 'SW', 'NEM', 'MOH', 'SMCI', 'CLX', 'EQT', 'FSLR', 'CBOE', 'K', 'NFLX', 'DLTR', 'WST', 'OXY', 'DPZ', 'SJM', 'GEN', 'LLY']
    # tickers = ["SPX", "NDX"]
    # all_tickers = ['MRNA', 'COP', 'MOH', 'AXON', 'WST', 'EQT', 'AZO', 'DXCM', 'DELL', 'EXR', 'ENPH', 'BG', 'CRWD',
    #               'EPAM', 'OXY',
    #               'TGT', 'TSLA', 'CF', 'ANET', 'TSCO']

    # tickers = ['MRNA', 'TGT', 'CF', 'AXON', 'EXR', 'CRWD', 'DXCM', 'TSLA', 'ANET', 'OXY']
    # ret_full = gather_data(tickers)

    # ret = ret_full[len(ret_full) - 500 - T:]
    # The regressor at time t: X[t] has to be constructed such it predicts y[t]
    # We construct it before the simulation run to keep everything tidy and easy to check
    # p = 4
    # X = indicators.build_reduced_regressor(ret, p) # after structure estimation
    # X = indicators.build_regressor(ret, vol, p) # before structure estimation

    ## set y = ret[p+1:], where p is the lag in AR(p), then X[t] is regressor for y[t], P@X[t] = y_hat for y[t]
    # y = ret[p+1:]

    # N = y.shape[1]
    # rho = X.shape[1]

    ## TOY EXAMPLE
    np.random.seed(0)
    rho = 5  # number of predictors
    N = 3  # number of response dimensions
    T = 250  # sample size

    # Beta becomes a matrix (rho x N)
    beta_true = np.random.randn(rho, N)

    # X stays the same: (T x rho)
    X = np.random.randn(T, rho) * 1/100

    noise_scale = np.linspace(0.001, 0.005, T).reshape(-1, 1)  # shape (T, 1)
    noise = noise_scale * np.random.randn(T, N)  # shape (T, N)

    # Multivariate response
    y = X @ beta_true + noise

    # 2) Collect \bar{V} and construct (V_0, delta_0), select priors alpha_0, beta_0, gamma_0

    V0, G0, delta0 = func.opt_prior(y, X, t_bar, N, rho, mu)

    #
    # Vyy0 = V0[:N, :N]
    # Vyx0 = V0[:N, N:] # in text: Vyx = Vxy.T
    # Vxx0 = V0[N:, N:]
    #
    # # # 3) Gather more data and run structure estimation
    # #
    # V = V0.copy()
    # for t in range(t_bar, T):
    #     d = np.concatenate((y[t], X[t]))
    #     V += np.outer(d, d)
    # Vyy = V[:N, :N]
    # Vyx = V[:N, N:]
    # Vxx = V[N:, N:]
    # Lambda = Vyy - Vyx @ np.linalg.inv(Vxx) @ Vyx.T
    # r, l_max = func.genetic_algorithm(Vyy, Vyy0, Vyx, Vyx0, Vxx, Vxx0, T-t_bar, delta0, mu, batch, p_m, m_iter, decay)
    # print(f"best parent:{r}, with likelihood:{l_max}")

    # 4) Run the model

    T_terminal = len(y)
    max_dd, variance_r, average_r, sign_acc = model_run(t_bar, T_terminal, y, X, V0, G0, delta0, alpha0, beta0, gamma0, phi0, N, rho, tickers)

    # 5) Evaluate

    print("max draw-down: ", max_dd)
    print("variance in portfolio returns: ", variance_r)
    print("average reward: ", average_r)
    print("sign accuracy: ", sign_acc/(T_terminal - T))
    print("---------------------------")

    # for i in range(len(tickers)):
    #     eval.plot_perf(y[:, i], tickers[i])

    print("FINISHED")