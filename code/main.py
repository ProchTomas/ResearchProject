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
        array of returns
    """
    stocks = ["aaon", "ait", "fix", "insm", "itci"]
    s = []
    stocks_returns = [f'c:/Users/TomasProchazka/PythonProjects/ResearchProject/stock_data/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers[:100]))
    s = np.column_stack(s)
    return s


def build_regressor(data1, t):
    """Help function to build a regressor
    The components of the regressor are predefined, optionally add data to the regressor
    Args:
    data1: returns
    y: dimension of the regressor
    t: time
    Retruns: Full regressor at time t, as definded
    """
    # Get pseudo-optimal actions up to time t
    
    regressor = np.array(data1[t-2:t+1]) 
    regressor = regressor.flatten()
    return regressor


def model_run(data, z_init, V_init, phi, nu, rho):
    """Runs the model on past data
    Args:
        data: data set
        z: initial regressor
        V0: initial covariance matrix
        phi: forgetting factor
        nu: data dimension
        rho: regressor dimension

    Returns:
        tuple: predictions, residuals
    """
    z_t = z_init
    L = np.linalg.inv(V_init)
    L_init = L
    
    residuals = []
    predictions = []
    
    ma_exp = 0
    
    for t in range(3, 50):
        # gather data to update L
        
        #ma_exp = indicators.mae(data[t], ma_exp, 2)
        optimal_actions = act.find_optimal_actions(data[:t+1], 0.002, 5)
        d_t = np.hstack((optimal_actions[t], z_t))
        # update L
        L = func.refill(L, d_t, phi)
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)

        # create the regressor
        z_t1 = build_regressor(optimal_actions, t)
        
        # calculate prediction
        pred = - np.linalg.inv(Lf).T @ Lzf.T @ z_t1
        
        pred = pred / pred.sum() # Normalization for action prediction
        
        e_hat = pred - data[t+1]
        predictions.append(pred)
        residuals.append(e_hat)
        z_t = z_t1
    
    # structure_estimation = func.genetic_algorithm(L_init, L, nu, rho, t)
    # print(f"Maximum likelihood of {structure_estimation[1]} has been reached with: {structure_estimation[0]}")
    
    return np.array(predictions), np.array(residuals)

if __name__ == "__main__":
    nu = 5
    rho = 15
    v = np.eye(nu + rho)*1
    tr_cost = 0.002
    
    data_set = gather_data()
    
    reg_0 = build_regressor(data_set, 3)
    
    model_results = model_run(data_set, reg_0, v, 0.95, nu, rho)
    
    pred_array = model_results[0]
    # print(pred_array)
    # res_array = model_results[1]
    # print(res_array)

    # opt_act = act.find_optimal_actions(data_set[:12], 0.002, nu)
    # print(f"Optimal actions: {opt_act}")
    # print(f"Rewards for my actions: {eval.reward(pred_array, data_set[2:len(pred_array)+2], tr_cost)}")
    even_actions = np.ones((len(pred_array), nu))*0.2
    # print(f"Rewards for even actions: {eval.reward(even_actions, data_set[2:len(pred_array)+2], tr_cost)}")
    eval.plot_reward(pred_array, data_set[3: len(pred_array)+3], tr_cost)
    
    #eval.residuals_time_plots(res_array)
    #eval.qq_plots(res_array)
    #print(np.round(opt_act))
    
    
    # SIMPLE EXAMPLE
    # states = np.array([[-0.0913041 , -0.08946959],
    # [ 0.06654758,  0.02052656],
    # [-0.03359455,  0.01767233],
    # [ 0.05817014, -0.09328192],
    # [ 0.04623629, -0.0128111 ],
    # [ 0.01294508,  0.08846527],
    # [ 0.00631335,  0.01738005],
    # [-0.02190309, -0.05807255],
    # [ 0.04788694, -0.02657513],
    # [ 0.05346319,  0.04080269]])
