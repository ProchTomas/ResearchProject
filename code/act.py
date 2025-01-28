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


def predicted_action():
    x = np.hstack()
    
    
    
    pass