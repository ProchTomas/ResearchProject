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
    """
    
    # x_t = Ax_t-1
    A = np.zeros((2*N+rho, 2*N+rho))
    A[N:2*N, :N] = np.eye(N)
    A[3*N:, 2*N:N+rho] = np.eye(rho - N)

    # a_t = Ba_t-1
    B = np.zeros((2*N + rho, N))
    B[:N, :N] = np.eye(N)

    # D is a cost matrix
    D = np.eye(N) * tr_costs
    
    # V(x_t) = x_t^T P x_t
    P = np.zeros((2*N+rho, 2*N+rho))
    P[:N, :N] = D
    P[:N, N:2*N] = -1*D
    P[N:2*N, :N] = -1*D
    P[N:2*N, N:2*N] = D
    P[2*N:3*N, :N] = -1*np.eye(N)
    
    return A, B, D, P

    
def update_A(N, rho, A_hat_t, A):
    """
    Args:
        N: number of assets
        rho: dimension of the regressor
        A_hat_t: estimate for the covariance matrix
        A: x updating matrix
    Returns: updated matrix A
    """
    A[2*N:3*N, 2*N:2*N + rho] = A_hat_t
    return A


def action_generation(x, A, P, B, D):
    """Generate optimal action
    Args:
        x: state vector
        A: x updating matrix
        P: state reward matrix
        B: action updating matrix
        D: cost matrix
    Returns: optimal action for current state
    """
    opt_act = - x @ A.T @ P @ B @ np.linalg.inv(D)
    
    # Normalize to obtain viable action
    return opt_act / np.sum(opt_act)

# Test functionality
N = 2
rho = 4
tr_costs = 0.002

A_hat = np.ones((N, rho)) * 0.4 # Set A_hat ambiguous
A, B, D, P = initialize_matrices_for_Ricatti_reccursion(N, rho, tr_costs)
A = update_A(N, rho, A_hat, A)
x = np.array([0.35, 0.65, 0.2, 0.8, -0.03, -0.1, 0.23, 0.09])
optimal_action = action_generation(x, A, P, B, D)
