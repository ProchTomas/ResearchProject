import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import func # utility functions for the model
import indicators # indicator functions
import eval # evaluation functions


def gather_data():
    """ Gather the data: returns for the assets

    Returns:
        array of returns
    """
    s = []
    stocks = ['returns.txt', 'vzreturns.txt']
    for stock in stocks:
        numbers = []
        with open(stock, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers[:100]))
    s = np.column_stack(s)
    return s


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


def run_simulation(z, V0, MSE, phi):
    z_t = z
    L = V0
    residuals = []
    predictions = []
    for t in range(6, len(states) - 1):
        # gather data to update L
        d_t = np.hstack((states[t], z_t))
        # update L
        L = func.refill(L, d_t, phi)
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)

        # create the regressor
        z_t1 = states[t - 5:t+1]
        z_t1 = z_t1.flatten()
        
        # calculate prediction
        pred = -np.linalg.inv(Lf).T @ (Lzf.T @ z_t1)
        e_hat = pred - states[t+1]
        predictions.append(pred)
        residuals.append(e_hat)
        z_t = z_t1
    return predictions, residuals


if __name__ == "__main__":
    states = gather_data()
    nu = 2
    rho = 12
    V0 = np.eye(nu + rho)*1e-4
    z = np.array(states[0:6])
    z = z.flatten()
    #z_t = z_t.flatten()