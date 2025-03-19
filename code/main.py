import math
import numpy as np
import matplotlib.pyplot as plt
import act # functions regarding action generation
import func # utility functions for the model
import indicators # indicator functions
import eval # evaluation functions


def gather_data():
    """ Gather the data

    Returns:
        returns and volumes to market cap
    """
    stocks = ["AAPL", "AMZN", "AVGO", "GOOGL", "MSFT"]
    s = []
    stocks_returns = [f'c:/Users/TomasProchazka/PythonProjects/ResearchProject/stock_data/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers[:300]))
    s = np.column_stack(s)
    
    v = []
    stocks_volumes = [f'c:/Users/TomasProchazka/PythonProjects/ResearchProject/stock_data/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    for stck_vol in stocks_volumes:
        numbers = []
        with open(stck_vol, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        v.append(np.array(numbers[:300]))
    v = np.column_stack(v)
    
    return s, v


def save_objects(matrix, best_regressor_subset):
    np.save('c:/Users/TomasProchazka/PythonProjects/ResearchProject/saved/regression_matrix.npy', matrix)
    np.save('c:/Users/TomasProchazka/PythonProjects/ResearchProject/saved/best_regressor_subset.npy', best_regressor_subset)


def get_indicator_values(data1, data2, time, mae_old):
    """Help function to get the current useful indicator values

    Args:
        data1: returns
        data2: volumes
        time: current time
        mae_old: old moving average exponential values

    Returns:
        array of indicator values
    """
    indicator_values = np.array(indicators.sma(time, data1, 21))
    indicator_values = np.append(indicator_values, indicators.rsi(time, data1, 21))
    
    # mae_new_1 = indicators.mae(data1[time], mae_old, 21)
    # mae_new_2 = indicators.mae(data1[time], mae_old, 7)
    
    # indicator_values = np.append(indicator_values, mae_new_1)
    # indicator_values = np.append(indicator_values, indicators.macd(mae_new_2, mae_new_1))
    
    # indicator_values = np.append(indicator_values, indicators.stoch_osc(time, data1, 21))
    # indicator_values = np.append(indicator_values, indicators.vol_osc(time, data2, 7, 21))
    return indicator_values


def build_regressor(data_returns, data_volumes, t, mae):
    """Help function to build a regressor
    The components of the regressor are predefined, optionally add data to the regressor
    Args:
        data_returns: returns
        data_volumes: volumes
        t: time
        mae: current value of moving average exponential
    Retruns: Full regressor at time t, as definded
    """
    regressor = np.array(data_returns[t-8:t+1])
    regressor = np.append(regressor, data_volumes[t-8:t+1])
    regressor = np.append(regressor, get_indicator_values(data_returns, data_volumes, t, mae))
    regressor = np.append(regressor, 1) 
    regressor = regressor.flatten()
    # Remove certain values (structure estimation)
    regressor = np.delete(regressor, [-2, -4])
    return regressor


def model_run(data_r, data_v, z_init, V_init, phi, nu, rho, t_cost):
    """Runs the model on past data
    Args:
        data_r: data set returns
        data_v: data set volumes
        z_init: initial regressor
        V_init: initial covariance matrix
        phi: forgetting factor
        nu: data dimension
        rho: regressor dimension
        t_cost: transaction cost
    Returns:
        tuple: predictions, residuals
    """
    z_t = z_init
    # L = np.linalg.inv(V_init)
    L = np.load('c:/Users/TomasProchazka/PythonProjects/ResearchProject/saved/regression_matrix.npy')
    reduction = np.load('c:/Users/TomasProchazka/PythonProjects/ResearchProject/saved/best_regressor_subset.npy')
    reduction = np.insert(reduction, 0, np.ones(nu))
    L = func.reduce_matrix(L, reduction)
    L = func.householder_transform(L)
    
    L_init = L
    
    X, Y, D, Q = act.initialize_matrices_for_Ricatti_reccursion(nu, rho, t_cost)
    # pick previous actions
    action_t = np.ones(nu) / nu
    action_prev = np.ones(nu) / nu
    
    residuals = []
    predictions = []
    actions = []
    
    mae_old = 0
    
    for t in range(99, 299):
        # gather data to update L
        
        # ma_exp = indicators.mae(data[t], ma_exp, 2)
        # optimal_actions = act.find_optimal_actions(data[:t+1], 0.002, 5)
        d_t = np.hstack((np.flip(data_r[t]), z_t))
        # update L
        
        phi = func.opt_forgetting_factor(z_t, func.getL_z(L, nu, rho), nu, phi_init=0.995)
        L = func.refill(L, d_t, phi)
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)

        # create the regressor
        z_t1 = build_regressor(data_r, data_v, t, mae_old)
        mae_now = indicators.mae(data_r[t], mae_old, 21)
        mae_old = mae_now
        
        # calculate prediction
        A_hat = - np.linalg.inv(Lf).T @ Lzf.T
        pred = A_hat @ z_t1
        
        e_hat = data_r[t+1] - pred
        predictions.append(pred)
        residuals.append(np.dot(e_hat, e_hat))
        z_t = z_t1
        
        # Actions
        x_t = np.hstack((action_t, action_prev, z_t))
        action_t = act.action_generation(nu, rho, x_t, X, Y, Q, A_hat, 7)
        action_prev = action_t
        actions.append(action_t)
    
    
    # structure_estimation = func.genetic_algorithm(L_init, L, nu, rho, t, 16)
    # print(f"Maximum likelihood of {structure_estimation[1]} has been reached with: {structure_estimation[0]}")
    # save_objects(L, structure_estimation[0])
    return np.array(predictions), np.array(residuals), np.array(actions)

if __name__ == "__main__":
    tr_cost = 0.002
    
    data_set_returns, data_set_volumes = gather_data()
    
    reg_0 = build_regressor(data_set_returns, data_set_volumes, 21, 0)
    nu = len(data_set_returns[0])
    rho = len(reg_0)
    v = np.eye(nu + rho)*1e-1
    
    model_results = model_run(data_set_returns, data_set_volumes, reg_0, v, 0.9, nu, rho, tr_cost)
    my_actions = model_results[2]
    print(my_actions)
    # pred_array = model_results[0]
    # res_array = model_results[1]
    # print(res_array)

    # opt_act = act.find_optimal_actions(data_set[:12], 0.002, nu)
    # print(f"Optimal actions: {opt_act}")
    print(f"Rewards for my actions: {eval.reward(my_actions, data_set_returns[100:len(model_results[2])+100], tr_cost)}")
    even_actions = np.ones((199, nu)) / len(data_set_returns[0])
    print(f"Rewards for even actions: {eval.reward(even_actions, data_set_returns[100:299], tr_cost)}")
    
    eval.plot_reward(my_actions, data_set_returns[100:299], tr_cost)
    # eval.plot_reward(even_actions, data_set[3: len(pred_array)+3], tr_cost)
    
    
    #eval.residuals_time_plots(res_array)
    #eval.qq_plots(res_array)
    #print(np.round(opt_act))
    