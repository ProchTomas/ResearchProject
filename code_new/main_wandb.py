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
    s = np.column_stack(s)
    
    v = []
    stocks_volumes = [f'data/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    for stck_vol in stocks_volumes:
        numbers = []
        with open(stck_vol, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        v.append(np.array(numbers))
    v = np.column_stack(v)
    
    return s, v



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

    sign_accuracy = np.array([0, 0])
    mse = 0
    residuals = []
    predictions = []
    actions = []
    a_prev = np.ones(N)/N
    actions.append(a_prev)

    D = 0.002*np.eye(N)
    A, B, X = act.initialize_evolution_matrices(N, rho)
    rewards = [1]

    omega_arr = []

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
        omega_arr.append(omega)
        # print("omega: ", omega)
        Q_sqrt = act.get_loss_matrix(N, rho, D, sigma, omega)
        A = act.update_evolution_matrix(N, rho, P, X, A)
        s_t = np.hstack((actions[-1], np.zeros(N), y_hat, x[t]))
        a_t = act.action_generation(N, rho, s_t, B, Q_sqrt, A, 10, P, Gxx, np.linalg.inv(sigma), False)
        actions.append(a_t)
        # print("action for: ", y[t], "is: ", a_t, "estimated y: ", y_hat)
        tr_c = 0
        for i in range(N):
            tr_c += np.abs(actions[-1][i] - a_t[i])
        reward = (1+a_t@y[t])*(1-tr_c)
        rewards.append(rewards[-1]*reward)
    # print(rewards)
    # plt.plot(rewards)
    # plt.show()
    # plt.plot(omega_arr)
    # plt.show()
    m_dd, var_r, avg_r = eval.risk_metrics(rewards)
    # print("max draw-down, variance in returns, average return", eval.risk_metrics(rewards))
    # print("sign accuracy: ", sign_accuracy/(terminal_time-t_0))
    return m_dd, var_r, avg_r


if __name__ == "__main__":
    # 0) Select optional parameters
    t_bar = 30  # time taken to collect V_bar

    ##  params in prior
    # mu = 0.8 # Shrinkage parameter for the prior (V_0, delta_0)
    ##  forgetting params
    # alpha0 = 0.009
    # beta0 = 0.9
    # gamma0 = 1 - alpha0 - beta0
    sweep_config = {
        "method": "random",  # or 'bayes'
        "metric": {"name": "m_dd", "goal": "minimize"},
        "parameters": {
            "mu": {"distribution": "uniform", "min": 0.01, "max": 0.99},
            "alpha0": {"distribution": "uniform", "min": 0.01, "max": 0.99},
            "beta0": {"distribution": "uniform", "min": 0.01, "max": 0.99},
            # gamma0 will be computed as 1 - alpha0 - beta0
        }
    }

    # GA params
    batch = 4 # number of mutations in each iteration of the population
    p_m = 0.25 # mutation probability
    m_iter = 1e2 # maximum number of iterations of the GA
    decay = 0.95 # decay rate for p_m

    # 1) Collect arrays y and x
    # TODO: run on market data, ensure y, x have the same structure as test data\
    tickers = ["AAPL", "GOOGL"]
    ret_full, vol_full = gather_data(tickers)

    ret = ret_full[:600]
    vol = vol_full[:600]
    p = 4
    X = indicators.build_reduced_regressor(ret, vol, p) # X[t] has to be constructed such it predicts y[t]
    # set y = ret[p+1:], where p is the lag in AR(p), then X[t] is regressor for y[t], P@X[t] = y_hat for y[t]
    y = ret[p+1:]

    N = y.shape[1]
    rho = X.shape[1]

    # TOY EXAMPLE
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
    def train():
        with wandb.init() as run:
            config = wandb.config
            alpha0 = config.alpha0
            beta0 = config.beta0
            gamma0 = 1-alpha0-beta0
            if gamma0 < 0:
                run.summary["m_dd"] = None
                run.summary["var_r"] = None
                run.summary["avg_r"] = None
                return

            mu = config.mu
            V0, G0, delta0 = func.opt_prior(y, X, t_bar, N, rho, mu)

            Vyy0 = V0[:N, :N]
            Vyx0 = V0[:N, N:] # in text: Vyx = Vxy.T
            Vxx0 = V0[N:, N:]

    # 3) Gather more data and run structure estimation

    # V = V0.copy()
    # T = 20
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
            max_dd, variance_r, average_r = model_run(t_bar, T_terminal, y, X, V0, G0, delta0, alpha0, beta0, gamma0, N, rho)
            wandb.log({
                "m_dd": max_dd,
                "var_r": variance_r,
                "avg_r": average_r,
                "alpha0": alpha0,
                "beta0": beta0,
                "gamma0": gamma0,
                "mu": mu
            })
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, train, count=50)
    # 5) Evaluate


    print("FINISHED")