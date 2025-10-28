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
    V_naive = V_prior.copy()
    V_simple = V_prior.copy()

    delta_t = delta_0
    Gyy_0 = G_0[:N, :N]
    Gyx_0 = G_0[:N, N:]
    Gxx_0 = G_0[N:, N:]

    sign_accuracy = np.zeros(N)
    # mse = 0
    residuals = []
    # predictions = []

    n_sim = 50

    actions = []
    a_prev = np.ones(N)/N
    # a_even = a_prev.copy()
    actions = [[a_prev] for _ in range(n_sim)]

    actions_agg = actions.copy()
    # actions_even = []
    # actions_even.append(a_prev)

    D = 0.002*np.eye(N)
    Q_sqrt_greedy = act.get_loss_matrix_greedy(N, rho, D, 0, 0)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [[1] for _ in range(n_sim)]
    rewards_agg = [[1] for _ in range(n_sim)]
    # rewards_even = [1]

    # omega_arr = []
    # forget_params = []

    for t in range(t_0, terminal_time):
        if t%10 == 0:
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
        residuals.append(e_t)
        # predictions.append(y_hat)
        for j in range(N):
            if np.sign(y_hat[j]) == np.sign(y[t][j]):
                sign_accuracy[j] += 1

        # mse += np.dot(e_t, e_t)
        # 2) Optimize allocation

        sigma = 1 / delta_t * Gyy @ Gyy.T
        A = act.update_evolution_matrix(N, rho, P, X, A)

        for k in range(n_sim):
            omega = func.get_omega(D, sigma, actions[k][-1])
            # omega_arr.append(omega)
            Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
            s_t = np.hstack((actions[k][-1], np.zeros(N), y_hat, x[t]))
            # print("time", t, "k", k, "len actions k" , len(actions[k]))

            a_t = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 20,  P, Gxx, np.linalg.inv(sigma), sampling=True)

            a_t_greedy = act.action_generation(N, rho, s_t, B, Q_sqrt_greedy, A, 20,  P, Gxx, np.linalg.inv(sigma), sampling=True)
            actions_agg[k].append(a_t_greedy)
            actions[k].append(a_t)
            # actions_even.append(a_even)
            # print("action for: ", y[t], "is: ", a_t, "estimated y: ", y_hat)
            tr_c = 0
            tr_c_agg = 0
            for i in range(N):
                tr_c += D[i][i] * np.abs(actions[k][-1][i] - a_t[i])
                tr_c_agg += D[i][i] * np.abs(actions_agg[k][-1][i] - a_t[i])
            reward = (1 + a_t @ y[t]) * (1 - tr_c)
            reward_agg = (1 + a_t_greedy @ y[t]) * (1 - tr_c_agg)
            # reward_even = (1 + a_even @ y[t])
            rewards[k].append(rewards[k][-1] * reward)
            # print("time", t, "len reward k", k, "is", len(rewards[k]))
            rewards_agg[k].append(rewards_agg[k][-1] * reward_agg)
            # rewards_even.append(rewards_even[-1] * reward_even)
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

    max_draw_downs = []
    variances = []
    avg_returns = []

    max_draw_downs_agg = []
    variances_agg = []
    avg_returns_agg = []
    for k in range(n_sim):
        m_dd, var_r, avg_r = eval.risk_metrics(rewards[k])
        m_dd_agg, var_r_agg, avg_r_agg = eval.risk_metrics(rewards_agg[k])
        max_draw_downs.append(m_dd)
        variances.append(var_r)
        avg_returns.append(avg_r)
        max_draw_downs_agg.append(m_dd_agg)
        variances_agg.append(var_r_agg)
        avg_returns_agg.append(avg_r_agg)

    print("maximal max drawdown is", np.max(max_draw_downs), "minimum drawdown is", np.min(max_draw_downs))
    print("median max drawdown is", np.median(max_draw_downs))
    print("mean max drawdown is", np.mean(max_draw_downs))
    print("maximal variance is", np.max(variances), "minimum variance is", np.min(variances))
    print("median variance is", np.median(variances))
    print("mean variance is", np.mean(variances))
    print("maximal average return", np.max(avg_returns), "minimum average return", np.min(avg_returns))
    print("median average return", np.median(avg_returns))
    print("mean average return", np.mean(avg_returns))
    print("---------------------------")
    print("maximal max drawdown risk-seeking", np.max(max_draw_downs_agg), "minimum drawdown risk-seeking", np.min(max_draw_downs_agg))
    print("median max drawdown risk-seeking", np.median(max_draw_downs_agg))
    print("mean max drawdown risk-seeking", np.mean(max_draw_downs_agg))
    print("maximal variance risk-seeking", np.max(variances_agg))
    print("median variance risk-seeking", np.median(variances_agg))
    print("mean variance risk-seeking", np.mean(variances_agg))
    print("maximal average return risk-seeking", np.max(avg_returns_agg), "minimum average return risk-seeking", np.min(avg_returns_agg))
    print("median average return risk-seeking", np.median(avg_returns_agg))
    print("mean average return risk-seeking", np.mean(avg_returns_agg))
    print("---------------------------")
    eval.plot_simulation_wealth_paths(np.array(rewards))
    eval.plot_simulation_wealth_paths(np.array(rewards_agg))
    # print("max draw-down (uniform): ", m_dd_even)
    # print("variance in portfolio returns (uniform): ", var_r_even)
    # print("average reward (uniform): ", avg_r_even)
    # print("---------------------------")
    # eval.compare_rewards([rewards, rewards_agg, rewards_even], ["risk-averse", "risk-seeking", "uniform"])
    # eval.plot_omega(omega_arr)
    # eval.plot_forgetting_params(forget_params)
    # eval.average_allocation_chart(np.array(actions), tckrs, 10)
    # eval.average_allocation_chart(np.array(actions_agg), tckrs, 10)

    return 0, 0, 0, 0


if __name__ == "__main__":
    # TODO: SAMPLING IS RANDOM: DO SIMULATION STUDY
    # 0) Select optional parameters
    t_bar = 100  # time taken to collect V_bar
    T = 100 # time taken to collect more data for structure estimation

    ##  ==== params in prior ====
    mu = 0.8 # [0.7, 0.99] # set higher for sampling
    ##  forgetting params
    alpha0 = 0.3 # [0.02, 0.3]
    beta0 = 0.6 # [0.2, 0.6]
    gamma0 = 1 - alpha0 - beta0 # [0.2, 0.4]

    # ==== GA params ====
    batch = 16 # number of mutations in each iteration of the population
    p_m = 0.2 # mutation probability
    m_iter = 1*1e2 # maximum number of iterations of the GA
    decay = 0.99 # decay rate for p_m

    # 1) Collect arrays y and x
    # tickers = ["AAPL", "GOOGL"] # test1 lag = 4
    # tickers = ["SPX", "NDX", "GLD"] # test2 lag = 4
    # tickers = ["AAON", "AIT", "MSTR", "SMCI"] # test3 lag = 10
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT"] # test4 lag = 4
    ret_full, vol_full = gather_data(tickers)

    ret = ret_full[:500]
    vol = vol_full[:500]
    # The regressor at time t: X[t] has to be constructed such it predicts y[t]
    # We construct it before the simulation run to keep everything tidy and easy to check
    p = 4
    X = indicators.build_reduced_regressor(ret, vol, p) # after structure estimation
    # X = indicators.build_regressor(ret, vol, p) # before structure estimation

    # set y = ret[p+1:], where p is the lag in AR(p), then X[t] is regressor for y[t], P@X[t] = y_hat for y[t]
    y = ret[p+1:]

    N = y.shape[1]
    rho = X.shape[1]

    ## TOY EXAMPLE
    # np.random.seed(0)
    # rho = 5  # number of predictors
    # N = 3  # number of response dimensions
    # T = 50  # sample size
    #
    # # Beta becomes a matrix (rho x N)
    # beta_true = np.random.randn(rho, N)
    #
    # # X stays the same: (T x rho)
    # X = np.random.randn(T, rho) * 1/100
    #
    # noise_scale = np.linspace(0.001, 0.005, T).reshape(-1, 1)  # shape (T, 1)
    # noise = noise_scale * np.random.randn(T, N)  # shape (T, N)
    #
    # # Multivariate response
    # y = X @ beta_true + noise

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
    max_dd, variance_r, average_r, sign_acc = model_run(T, T_terminal, y, X, V0, G0, delta0, alpha0, beta0, gamma0, N, rho, tickers)

    # 5) Evaluate

    print("max draw-down: ", max_dd)
    print("variance in portfolio returns: ", variance_r)
    print("average reward: ", average_r)
    print("sign accuracy: ", sign_acc/(T_terminal - T))
    print("---------------------------")

    for i in range(len(tickers)):
        eval.plot_perf(y[:, i], tickers[i])

    print("FINISHED")