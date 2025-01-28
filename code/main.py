import math
import numpy as np
import matplotlib.pyplot as plt
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


def model_run(data, z_init, V_init, phi, nu, rho):
    """Calculated predictions for the next data

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
    residuals = []
    predictions = []
    
    ma_exp = 0
    
    for t in range(1, len(data) - 1):
        # gather data to update L
        
        #ma_exp = indicators.mae(data[t], ma_exp, 2)
        
        d_t = np.hstack((data[t], z_t))
        # update L
        L = func.refill(L, d_t, phi)
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)

        # create the regressor
        z_t1 = data[t]
        z_t1 = z_t1.flatten()
        
        # calculate prediction
        pred = - np.linalg.inv(Lf).T @ Lzf.T @ z_t1
        e_hat = pred - data[t+1]
        predictions.append(pred)
        residuals.append(e_hat)
        z_t = z_t1
    
    return np.array(predictions), np.array(residuals)

if __name__ == "__main__":
    nu = 2
    rho = 2
    v = np.eye(nu + rho)*1e-4
    tr_cost = 0.002
    
    states = np.array([[-0.0913041 , -0.08946959],
    [ 0.06654758,  0.02052656],
    [-0.03359455,  0.01767233],
    [ 0.05817014, -0.09328192],
    [ 0.04623629, -0.0128111 ],
    [ 0.01294508,  0.08846527],
    [ 0.00631335,  0.01738005],
    [-0.02190309, -0.05807255],
    [ 0.04788694, -0.02657513],
    [ 0.05346319,  0.04080269]])
    
    z = np.array(states[0])
    z = z.flatten()
    
    model_results = model_run(states, z, v, 0.95, nu, rho)
    
    pred_array = model_results[0]
    res_array = model_results[1]
    

    
    #opt_act = act.find_optimal_actions(states, 0.002, 2)
    #print(eval.reward(opt_act, states, tr_cost))
    #eval.plot_reward(opt_act, states, tr_cost)
    #eval.residuals_time_plots(res_array)
    #eval.qq_plots(res_array)
    #print(np.round(opt_act))
