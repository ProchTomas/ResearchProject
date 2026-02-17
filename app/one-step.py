import numpy as np
import act
import func
import indicators
import eval
import time
from numpy.linalg import det as det
from scipy.linalg import cholesky, qr, solve_triangular


def load_statistics():
    """
    :return:
        g: (N+rho, N+rho)-matrix
        h: (3N+rho+1, 3N+rho+1)-matrix
        delta: scalar
        a: N-vector
    """
    g = 1e-2*np.eye(4)
    h = 1e-1*np.eye(8)
    delta = 5
    a = np.array([0.1, 0.9])
    return g, h, delta, a

def get_regressor():
    return np.array([1, 2])

def get_observation():
    return np.array([1, 1.5])


def model_run(params, g, x, delta, a_, H_, n):
    mu, alpha0, beta0, phi0, D, g_0, delta0 = params.values()
    rho = len(x)
    A, B, X_mat = act.initialize_evolution_matrices(n, rho)

    gyy = g[:n, :n]
    gyx = g[:n, n:]
    gxx = g[n:, n:]

    P = gyx @ np.linalg.inv(gxx)
    y_hat = P @ x

    sigma = 1/delta * (gyy @ gyy.T)
    omega = func.get_omega(D, sigma, a_)

    q = act.get_loss_matrix(n, rho, D, sigma, omega)
    s = np.hstack((a_, np.zeros(n), y_hat, x))
    A_update = act.update_evolution_matrix(n, rho, P, X_mat, A)

    a, H = act.action_generation(n, rho, s, B, q, A_update, 1, H_, P, gxx, np.linalg.inv(sigma), sampling=False)

    return a, H


def update(params, g, y, x, delta, h_, h):
    mu, alpha0, beta0, phi_0, D, g_0, delta0 = params.values()
    gamma0 = 1-alpha0-beta0
    n = len(y)

    gyy_0 = g_0[:n, :n]
    gxx_0 = g_0[n:, n:]

    gyy = g[:n, :n]
    gyx = g[:n, n:]
    gxx = g[n:, n:]

    P = gyx @ np.linalg.inv(gxx)
    y_hat = P @ x
    e_hat = y - y_hat

    d = np.concatenate((y, x))
    args = (alpha0, beta0, gamma0, delta0 + 1, delta0, gxx, det(gxx), gxx_0, det(gxx_0), gyy, det(gyy), gyy_0,
            det(gyy_0), g @ g.T, g @ g.T + np.outer(d, d), g_0 @ g_0.T, gyy @ gyy.T,
            gyy @ gyy.T + 1 / (1 + x.T @ np.linalg.inv(gxx @ gxx.T) @ x) * e_hat @ e_hat.T,
            gyy_0 @ gyy_0.T, x.T @ np.linalg.inv(gxx @ gxx.T) @ x,
            e_hat.T @ np.linalg.inv(gyy @ gyy.T) @ e_hat)

    alpha, beta, gamma = func.opt_forget_factors(args)
    g_update = func.update_G(g, d, g_0, alpha, beta)
    delta_update = (alpha+beta)*delta + beta + gamma*delta0

    h_update = func.optimize_H(phi_0, h_, h)

    return g_update, delta_update, h_update


if __name__ == '__main__':
    model_parameters = {
        'mu': 0.9,
        'alpha0': 0.09,
        'beta0': 0.05,
        'phi0': 0.1,
        'D': 2e-4*np.eye(2),
        'G_0': 1e-3*np.eye(4),
        'delta0': 14,
    }
    tickers = []
    N = 2
    G, H, Delta, a_prev = load_statistics()

    X = get_regressor()

    opt_a, H_new = model_run(model_parameters, G, X, Delta, a_prev, H, N)

    Y = get_observation()

    G_update, Delta_update, H_update = update(model_parameters, G, Y, X, Delta, H, H_new)