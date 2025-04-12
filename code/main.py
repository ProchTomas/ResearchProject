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
    # stocks = ["DXY", "SPX"]
    stocks = ["GOOGL", "AAPL", "NVDA", "MSFT", "AMZN"]
    # stocks = ["AIT", "AAON", "FIX", "INSM", "ITCI", "LNW", "MSTR", "SFM", "SMCI", "UFPI"]
    # stocks = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "JNJ", "WMT", "JPM", "PG", "VZ"]
    # stocks = ["GS", "UNH", "MSFT", "HD", "CAT", "SHW", "CRM", "AMGN", "AXP", "MCD", "PEP", "JPM", "IBM", "TRV", "AAPL", "AMZN", "HON", "BA", "PG", "CVX", "MMM", "JNJ", "NVDA", "DIS", "MRK", "WMT", "NKE", "CSCO", "VZ"]
    s = []
    stocks_returns = [f'stock_data/{stck}_returns.txt' for stck in stocks]
    for stck_rtrn in stocks_returns:
        numbers = []
        with open(stck_rtrn, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        s.append(np.array(numbers))
    s = np.column_stack(s)
    
    v = []
    stocks_volumes = [f'stock_data/{stck}_volumes_to_mktcap.txt' for stck in stocks]
    for stck_vol in stocks_volumes:
        numbers = []
        with open(stck_vol, 'r') as file:
            for line in file:
                numbers.append(float(line.strip()))
        v.append(np.array(numbers))
    v = np.column_stack(v)
    
    return s, v


def save_objects(matrix, best_regressor_subset):
    np.save('saved/regression_matrix.npy', matrix)
    np.save('saved/best_regressor_subset.npy', best_regressor_subset)
    

def check_action_similarity(a_now, a_prev, epsilon=1e-2):
    """If the generated action is too similar to the previous action we won't change it,
    to save on transaction costs.
    """
    # Compute Euclidean distance of the two actions
    dist = np.linalg.norm(a_now - a_prev)
    if dist < epsilon:
        return a_prev
    else:
        return a_now


def get_indicator_values(data1, data2, time, mae_7, mae_21):
    """Helpful function to get the current indicator values

    Args:
        data1: returns
        data2: volumes
        time: current time
        mae_21: current moving average exponential values with period 21
        mae_7: current moving average exponential values with period 7

    Returns:
        array of indicator values
    """
    indicator_values = np.array(indicators.sma(time, data1, 10))
    indicator_values = np.append(indicator_values, indicators.rsi(time, data1, 10))

    # indicator_values = np.append(indicator_values, mae_21)
    # indicator_values = np.append(indicator_values, mae_7)
    
    # indicator_values = np.append(indicator_values, indicators.macd(mae_7, mae_21))
    
    # indicator_values = np.append(indicator_values, indicators.stoch_osc(time, data1, 21))
    # indicator_values = np.append(indicator_values, indicators.vol_osc(time, data2, 7, 21))
    return indicator_values


def build_regressor(data_returns, data_volumes, t, mae1, mae2):
    """Auxiliary function to build the regressor (uses the estimated structure)
    The components of the regressor are predefined, optionally add data to the regressor
    Args:
        data_returns: returns
        data_volumes: volumes
        t: time
        mae21: current value of moving average exponential with period 21
        mae7: current value of moving average exponential with period 7
    Retruns: Full regressor at time t, as definded
    """
    regressor = np.array(data_returns[t-9:t+1])
    regressor = np.append(regressor, data_volumes[t-9:t+1])
    regressor = np.append(regressor, get_indicator_values(data_returns, data_volumes, t, mae1, mae2))
    regressor = np.append(regressor, 1)
    regressor = regressor.flatten()
    return regressor


def model_run(data_r, data_v, z_init, V_init, phi, nu, rho, t_cost, start, end):
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
        start: starting point for the test
        end: end point for the test
    Returns:
        tuple: predictions, residuals
    """
    z_t = z_init
    L = np.linalg.inv(V_init)
    # L = np.load('saved/initial_regression_matrix.npy') ## if the initial apriori matrix L is given
    # 
    ## If you run the structure estimation again for your matrix L and don't want to compute it again, you can run this reduction
    ## beware that if you have many different data as regressors, making the matrix X is not an easy task
    ## However it may be worth it
    #
    # reduction = np.load('saved/best_regressor_subset.npy')
    # reduction = np.insert(reduction, 0, np.ones(nu))
    # L = func.reduce_matrix(L, reduction)
    # L = func.householder_transform(L)
    
    L_init = L
    
    X, Y, D, Q = act.initialize_matrices_for_Ricatti_reccursion(nu, rho, t_cost, 10) # S = 10 stands for the number of past states we put into the regressor
    # insert previous actions
    action_t = np.ones(nu) / nu
    action_prev = np.ones(nu) / nu
    
    residuals = []
    predictions = []
    actions = []
    
    mae_old_21 = np.zeros(nu)
    mae_old_7 = np.zeros(nu)
    
    for t in range(start, end):
        # update L
        d_t = np.hstack((data_r[t], z_t))
        
        phi = func.opt_forgetting_factor(z_t, func.getL_z(L, nu, rho), nu, phi_init=0.95)

        L = func.refill(L, d_t, phi)
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)

        # mae_now_21 = indicators.mae(data_r[t], mae_old_21, 21)
        # mae_old_21 = mae_now_21
        # mae_now_7 = indicators.mae(data_r[t], mae_old_7, 7)
        # mae_old_7 = mae_now_7

        # create the regressor
        z_t1 = build_regressor(data_r, data_v, t, mae_now_7, mae_now_21)
        
        # calculate prediction
        A_hat = - np.linalg.inv(Lf).T @ Lzf.T
        pred = A_hat @ z_t1
        
        e_hat = data_r[t+1] - pred
        predictions.append(pred)
        residuals.append(e_hat)
        z_t = z_t1
        
        # actions
        X = act.update_X(nu, rho, A_hat, X)
        x_t = np.hstack((action_t, action_prev, z_t))
        
        action_t = act.action_generation(nu, rho, x_t, X, Y, Q, A_hat, 9)
        action_t = check_action_similarity(action_t, action_prev)
        
        action_prev = action_t
        actions.append(action_t)
    
    # structure_estimation = func.genetic_algorithm(L_init, L, nu, rho, t, 16)
    # print(f"Maximum likelihood of {structure_estimation[1]} has been reached with: {structure_estimation[0]}")
    # save_objects(L, structure_estimation[0])

    return np.array(predictions), np.array(residuals), np.array(actions)


def one_step_run(action_prev, action_before, data, z, update):
    """
    Args:
        action_prev: previous action
        action_before: action the day before (not neccessary, it will be replaced by the previous action)
        data: newly observed data to update L (this is done after observing the new state)
        z: regressor for the current state
        update: if True - updates the matrix L, if False

    Returns:
        Predicts the next state, outputs the prediction and optimal action for the next step
    """
    L = np.load('saved/regression_matrix.npy')
    
    if update == True:
        d_t = np.hstack((data, z))
        phi = func.opt_forgetting_factor(z, func.getL_z(L, nu, rho), nu, phi_init=0.95)
        L = func.refill(L, d_t, phi)
        np.save('saved/regression_matrix.npy', L)
        print("Matrix L has been updated and saved")
        pass
    
    else:
        Lf = func.getL_f(L, nu)
        Lzf = func.getL_zf(L, nu, rho)
        
        t_cost = 0.002
        X, Y, D, Q = act.initialize_matrices_for_Ricatti_reccursion(nu, rho, t_cost, 10)
        
        A_hat = - np.linalg.inv(Lf).T @ Lzf.T
        pred = A_hat @ z
        
        X = act.update_X(nu, rho, A_hat, X)
        x_t = np.hstack((action_prev, action_before, z))
        action_t = act.action_generation(nu, rho, x_t, X, Y, Q, A_hat, 9)
        action_t = check_action_similarity(action_t, action_prev)
        
        return pred, action_t
    

if __name__ == "__main__":
    tr_cost = 0.002
    
    data_set_returns, data_set_volumes = gather_data()
    end = len(data_set_returns) - 1
    start = end - 251 # one trading year
    
    nu = len(data_set_returns[0])
    reg_0 = build_regressor(data_set_returns, data_set_volumes, start, mae1=np.zeros(nu), mae2=np.zeros(nu))
    rho = len(reg_0)
    v = np.eye(nu + rho)
    
    model_results = model_run(data_set_returns, data_set_volumes, reg_0, v, 0.95, nu, rho, tr_cost, start, end)
    my_actions = model_results[2]

    pred_array = model_results[0]
    res_array = model_results[1]
    
    rewards_for_my_actions = eval.reward(my_actions, data_set_returns[start+1:len(my_actions)+start+1], tr_cost)
    print(f"Rewards for my actions: {rewards_for_my_actions}")
    
    even_actions = np.ones((end-start, nu)) / len(data_set_returns[0])
    rewards_for_even_actions = eval.reward(even_actions, data_set_returns[start+1:len(even_actions)+start+1], tr_cost)
    print(f"Rewards for even actions: {rewards_for_even_actions}")
    
    eval.plot_reward(my_actions, data_set_returns[start+1:len(my_actions)+start+1], tr_cost)
    print(f"Max drawdown for my actions: {eval.max_drawdown(rewards_for_my_actions)}")
    
    eval.plot_reward(even_actions, data_set_returns[start+1: len(even_actions)+start+1], tr_cost)
    print(f"Max drawdown for even actions: {eval.max_drawdown(rewards_for_even_actions)}")
    
    stocks = ["GOOGL", "AAPL", "NVDA", "MSFT", "AMZN"]
    eval.residual_plots(pred_array, res_array, stocks)
    eval.residuals_time_plots(res_array)
    eval.qq_plots(res_array, stocks)
    
    eval.average_allocation_chart(my_actions, stocks, 21)
    
    actions = [my_actions,  even_actions]
    labels = ["Algorithm", "Uniform"]
    eval.plot_multiple_rewards(actions, data_set_returns[start+1:len(my_actions)+start+1], 0.002, labels)
    
