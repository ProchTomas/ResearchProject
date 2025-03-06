import numpy as np
import math
from scipy.optimize import minimize


def utility_function(actions, data, costs):
    """ utility function for the find_optimal_actions function
    
    Args:
        actions: array of actions
        data: data array
        costs: transaction costs
    Returns:
        utility function value
    """
    utility = 0
    for t in range(1, len(actions)):
        transaction_cost = 0
        for i in range(len(actions[0])):
            transaction_cost += costs * np.abs(actions[t][i] - actions[t-1][i])
        utility += np.log((1 + data[t] @ actions[t]) * (1 + transaction_cost))
    return utility


def find_optimal_actions(data, cost, N):
    """
    Args:
        data: data array
        cost: transaction costs
        N: number of assets

    Returns:
        optimal actions array
    """
    num_actions = len(data)
    a0 = np.ones((num_actions, N)) / N

    def objective(a):
        actions = a.reshape((num_actions, N))
        return -utility_function(actions, data, cost)

    cons = [{'type': 'eq', 'fun': lambda a, i=i: np.sum(a[N*i:(i+1)*N]) - 1} for i in range(num_actions)]

    bounds = [(0, 1)] * (num_actions * N)
    result = minimize(objective, a0.flatten(), constraints=cons, bounds=bounds)
    return result.x.reshape((num_actions, N))


def initialize_matrices_for_Ricatti_reccursion(N, rho, tr_costs):
    """Function to construct matrices for the Ricatti reccursion

    Args:
        N: number of assets
        rho: dimension of the regressor
        tr_costs: transactions costs
    Returns:
        Initial matrices X, Y, D and the square root of loss matrix Q
    """
    
    # x_t = X_tx_t-1 + Ya_t
    # TODO initialize X properly after the structure of F is decided
    X = np.zeros((2*N+rho, 2*N+rho))
    X[N:2*N, :N] = np.eye(N)
    X[3*N:, 2*N:N+rho] = np.eye(rho - N)

    Y = np.zeros((2*N + rho, N))
    Y[:N, :N] = np.eye(N)

    # D is a cost matrix
    D = np.eye(N) * tr_costs
    D_sqrt = np.sqrt(D)
    D_sqrt_inv = np.linalg.inv(D_sqrt)
    
    # V(x_t) = x_t^T Q^TQ x_t
    Q = np.zeros((2*N+rho, 2*N+rho))
    Q[:N, :N] = D_sqrt
    Q[:N, N:2*N] = -1*D_sqrt
    Q[:N, 2*N:3*N] = -1/2*D_sqrt_inv
    
    return X, Y, D, Q

    
def update_X(N, rho, A_hat_t, X_now):
    """
    Args:
        N: number of assets
        rho: dimension of the regressor
        A_hat_t: estimate for the covariance matrix
        X_now: x updating matrix
    Returns: updated matrix X
    """
    # TODO include matrix F for regressors update
    X_new = X_now
    X_new[2*N:3*N, 2*N:2*N + rho] = A_hat_t
    return X_new


def action_generation(N, rho, x, X_now, Y, Q, A, h):
    """Generate optimal action
    Args:
        N: number of assets
        rho: dimension of the regressor
        x: state vector (row vector)
        X_now: state updating matrix
        Y: state updating matrix
        Q: square root of the loss matrix
        A: matrix of regression coefficients
        h: horizon
    Returns: optimal action for current state
    """

    # Initialize H
    H = np.zeros_like(Q)
    X = update_X(N, rho, A, X_now)

    # Iterate backwards in time
    for j in range(h):
        H_tilde = np.block([[Q], [H]])

        # Orthogonal transformation of (Q \\ H), U^TU = I
        # R is the new H_tilde
        U, R = np.linalg.qr(H_tilde)

        # Make the estimation and add row of zeros below the new matrix to make up for lost dimension
        H_YX = np.block([[R @ Y, R @ X]])

        # Get submatrices
        H_a = H_YX[:N, :N]
        H_x = H_YX[:N, N:]

        # Calculate Lagrange multiplier lambda
        ones = np.ones((N, 1))
        H_a_inv = np.linalg.inv(H_a)

        one_ = np.zeros_like(x)
        one_[-1] = 1

        M = np.outer(ones, one_ + ones.T @ H_a_inv @ H_x / (ones.T @ H_a_inv @ ones))
        H_new = np.block([[H_YX[N:, N:]], [M]])
        U_2, R_2 = np.linalg.qr(H_new)

        # Set H to H_new for the next iteration
        H = R_2
    # Get the optimal action
    lmbda = - (1 + ones.T @ H_a_inv @ H_x @ x.reshape(-1, 1)) / (ones.T @ H_a_inv @ ones)
    lmbda = lmbda.item()

    a_opt = - H_a_inv @ (H_x @ x.reshape(-1, 1) + lmbda * ones)
    a_opt = a_opt.flatten()
    return a_opt
    
    
    
    

n = 2
m = 2
x, y, d, q = initialize_matrices_for_Ricatti_reccursion(n, m, 0.02)
z = np.array([0.8, 0.2, 0.2, 0.8, -0.05, -0.01])

action_generation(n, m, z, x, y, q, 0, 2)


    
