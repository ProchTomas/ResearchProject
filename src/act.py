import numpy as np
import math
from scipy.optimize import minimize
from scipy.linalg import eigh, cholesky
import func


def initialize_evolution_matrices(n, rho):
    """Constructs evolution matrices
    :arg
        n: number of assets
        rho: dimension of regressor
    :return
        initial evolution matrices A, B, X
    """
    a = np.zeros((3*n+rho, 3*n+rho))
    b = np.zeros((3*n+rho, n))
    # TODO: properly initialize X (depends on the estimated regressor subset)
    x = np.eye(rho)

    a[n:2*n, :n] = np.eye(n)
    a[3*n:, 3*n:] = x
    b[:n, :n] = np.eye(n)

    return a, b, x

def get_loss_matrix(n, rho, d, sigma, omega):
    """Constructs the square root of the loss matrix Q

    :arg
        n: number of assets
        rho: dimension of the regressor
        d: cost matrix
        sigma: covariance matrix
        omega: multiplicative factor
    :returns
        Q^1/2: square root of the loss matrix (upper triangular matrix)
    """
    Q = np.zeros((3*n + rho, 3*n + rho), dtype=float)
    e_tilde = d + 0.5*omega**2*sigma
    e_tilde_inv = np.linalg.inv(e_tilde)
    f = e_tilde_inv @ d
    # WORKING VERSION q
    # q = np.block([
    #     [e_tilde, -d, -0.5 * omega * np.eye(n)],
    #     [-d, d, 0.5* omega * np.eye(n)],
    #     [-0.5 * omega * np.eye(n), 0.5 * omega * np.eye(n), 0.5 * (omega ** 2) * np.linalg.inv(d)]
    # ])
    epsilon = 1e-7*np.eye(n)
    q = np.block([
        [e_tilde + epsilon, -d, -0.5 * omega * np.eye(n)],
        [-d, epsilon + d @ f, 0.5 * omega * f.T],
        [-0.5 * omega * np.eye(n), 0.5 * omega * f, epsilon + 0.25 * (omega ** 2) * e_tilde_inv]
    ])
    l = cholesky(q, lower=True)  # ll^T = q, l^T is upper triangular
    r = l.T  # r^Tr = q
    R = np.zeros_like(Q)
    R[:3 * n, :3 * n] = r  # R^TR = Q, R is upper triangular
    return R

    # GREEDY STRATEGY
    # q = np.block([
    #     [np.sqrt(d) , -np.sqrt(d), -0.5 * np.linalg.inv(np.sqrt(d))],
    #     [np.zeros_like(d), np.zeros_like(d), np.zeros_like(d)],
    #     [np.zeros_like(d), np.zeros_like(d), np.zeros_like(d)]
    # ])
    #
    # Q[:3*n, :3*n] = q
    # return Q

def get_loss_matrix_greedy(n, rho, d, sigma, omega):
    """Constructs the square root of the greedy loss matrix Q

    :arg
        n: number of assets
        rho: dimension of the regressor
        d: cost matrix
        sigma: covariance matrix
        omega: multiplicative factor
    :returns
        Q^1/2: square root of the greedy loss matrix (upper triangular matrix)
    """
    Q = np.zeros((3 * n + rho, 3 * n + rho), dtype=float)
    q = np.block([
        [np.sqrt(d) , -np.sqrt(d), -0.5 * np.linalg.inv(np.sqrt(d))],
        [np.zeros_like(d), np.zeros_like(d), np.zeros_like(d)],
        [np.zeros_like(d), np.zeros_like(d), np.zeros_like(d)]
    ])

    Q[:3*n, :3*n] = q
    return Q

    
def update_evolution_matrix(n, rho, p, x, a):
    """
    :arg
        n: number of assets
        rho: dimension of the regressor
        p: point estimate of the regression coefficients matrix
        x: point estimates of the regressors evolution matrix
        a: current evolution matrix
    :returns
        updated evolution matrix
    """
    a_new = a.copy()
    a_new[2*n:3*n, 3*n:3*n+rho] = p
    if x is not np.empty:
        a_new[3*n:, 3*n:] = x

    return a_new


def smalbe_cqp_solver(H_a, H_x, x_state, N, lambda_init, a_initial_guess):
    """
    Solves the constrained quadratic program using the SMALBE method.
    Minimizes: 1/2 * a^T * (H_a^T H_a) * a + (x_state^T H_x^T H_a) * a
    Subject to: sum(a) = 1 and a >= 0
    """
    # TODO: check SUPREMELY WELL this aligns with the text

    def solve_bound_qp_projected_gradient(H_aug, b_aug, l_bound, a_start):
        """
        Inner solver for the bound-constrained QP using Projected Gradient method.
        Minimizes: 1/2 * a^T * (H_aug^T H_aug) * a - b_aug^T * a
        Subject to: a >= l_bound
        """
        action = np.maximum(l_bound, a_start)
        max_inner_iter = 250
        inner_tol = 1e-8

        for _ in range(max_inner_iter):
            g = H_aug.T @ (H_aug @ action) - b_aug
            # KKT conditions check for this sub-problem
            free_mask = action > l_bound
            kkt_viol_free = np.linalg.norm(g[free_mask])
            kkt_viol_active = np.linalg.norm(g[~free_mask & (g < 0)])
            if kkt_viol_free + kkt_viol_active < inner_tol:
                break

            # Projected gradient search direction
            d = -g
            d[ (action <= l_bound) & (g >= 0) ] = 0
            
            if np.linalg.norm(d) < 1e-12:
                break
            
            # Optimal step size for quadratic function
            g_dot_d = g.T @ d
            Hd = H_aug @ d
            d_H_d = Hd.T @ Hd
            
            if d_H_d <= 1e-12: # Zero or negative curvature
                # alpha = np.inf # original
                alpha = 10.0
            else:
                alpha = -g_dot_d / d_H_d

            # Max step to stay within bounds
            d_neg_mask = d < -1e-12
            if np.any(d_neg_mask):
                alpha_max = np.min((l_bound[d_neg_mask] - a[d_neg_mask]) / d[d_neg_mask])
                alpha = min(alpha, alpha_max)
            action -= alpha * g
            action = np.maximum(l_bound, a) # Project to handle precision errors
        return action

    # --- SMALBE main logic ---
    # QP parameters for: min 1/2 a^TAa - b^Ta
    b = -(H_a.T @ H_x @ x_state).flatten()

    # Constraint parameters: B a = c, a >= l_bound
    B = np.ones((1, N))
    c = np.array([1.0])
    l_bound = np.zeros(N)

    # SMALBE parameters
    xi = 1.0
    beta = 5.0
    lambda_ = lambda_init
    a = np.ones(N) / N if a_initial_guess is None else a_initial_guess.copy()
    a = np.maximum(l_bound, a) # Ensure initial guess is feasible
    
    max_outer_iter = 100
    outer_tol = 1e-7
    prev_constraint_violation_norm = np.inf

    for k in range(max_outer_iter):
        # Form augmented matrices for the inner problem using square-root formalism
        H_aug = np.vstack([H_a, np.sqrt(xi) * B])
        b_aug = b - (B.T @ lambda_).flatten() - xi * (B.T @ c).flatten()
        
        # Solve the bound-constrained inner problem
        a_new = solve_bound_qp_projected_gradient(H_aug, b_aug, l_bound, a_start=a)
        
        constraint_violation = B @ a_new - c
        constraint_violation_norm = np.linalg.norm(constraint_violation)
        
        # Check for convergence
        if np.linalg.norm(a_new - a) < outer_tol and constraint_violation_norm < outer_tol:
            a = a_new
            break

        # Update Lagrange multiplier
        lambda_ += xi * constraint_violation
        # print(f"iteration {k} lambda: {lambda_}")
        
        # Update penalty parameter
        if k > 0 and constraint_violation_norm > 0.5 * prev_constraint_violation_norm:
            xi *= beta

        a = a_new
        prev_constraint_violation_norm = constraint_violation_norm

    # Final projection to ensure constraints are met due to any floating point errors
    a = np.maximum(0, a)
    a /= np.sum(a)
    return a


def action_generation(N, rho, s, B, Q, A, h, H_init, p_hat, g_x, sigma_inv, sampling):
    """
    Generate optimal action using a constrained LQR framework solved by SMALBE
    :arg
        N: number of assets
        rho: dimension of the regressor
        s: state
        B: current evolution matrix for actions
        Q: loss matrix
        A: current evolution matrix
        h: terminal horizon
    :returns
        optimal action
    """
    # H = np.zeros_like(Q)
    H = H_init
    
    ones = np.ones((N, 1))
    e_N = np.zeros_like(s.T)
    e_N[-1] = 1

    H_a = np.zeros((N, N))
    H_x = np.zeros((N, 2*N+rho))

    for j in range(h):
        H_tilde = np.block([[Q], [H]])
        
        U, R = np.linalg.qr(H_tilde)
        if sampling:
            p = func.sample_matrix_normal(p_hat, sigma_inv, g_x)
            A = update_evolution_matrix(N, rho, p, np.empty, A)
        H_YX = R @ np.block([B, A])

        H_a_j = H_YX[:N, :N]
        H_x_j = H_YX[:N, N:]
        
        try:
            H_a_inv_j = np.linalg.inv(H_a_j)
        except np.linalg.LinAlgError:
            H_a_inv_j = np.linalg.pinv(H_a_j) # Use pseudo-inverse if singular
        
        M_numerator = e_N.T + (ones.T @ H_a_inv_j @ H_x_j)
        M_denominator = ones.T @ H_a_inv_j @ ones
        
        if np.abs(M_denominator) < 1e-12:
            M = np.zeros((1, H_x_j.shape[1])) # Avoid division by zero
        else:
            M = (1.0 / M_denominator.item()) * np.outer(ones, M_numerator)

        H_new = np.block([[H_YX[N:, N:]], [M]])
        _, R_2 = np.linalg.qr(H_new)
        H = R_2


    # After the backward recursion, set up the final QP problem for the current time
    H_tilde_final = np.block([[Q], [H]])
    _, R_final = np.linalg.qr(H_tilde_final)
    H_YX_final = R_final @ np.block([B, A])
    
    H_a = H_YX_final[:N, :N]
    H_x = H_YX_final[:N, N:]

    # Use the new SMALBE solver to find the optimal action
    # The state vector x is a row vector, convert to column for matrix math
    s_state_col = s.reshape(-1, 1)
    l = -np.linalg.inv(H_a)@H_x@s
    a_guess = np.exp(l)/np.sum(np.exp(l))
    if any(np.isnan(a_guess)):
        a_guess = s[:N]
    a_opt = smalbe_cqp_solver(H_a, H_x, s_state_col, N, M_denominator, a_guess)
    
    return a_opt, R_final


def markowitz_allocation(mu, cov, risk_aversion=1.0):
    """
    Computes the optimal Markowitz portfolio weights.

    :arg mu: 1D array of expected returns
    :arg cov: 2D array of the covariance matrix
    :arg risk_aversion: Lambda scalar balancing return vs. risk
    :returns: 1D array of optimal weights
    """
    n_assets = len(mu)

    def objective(w):
        # We minimize the negative utility
        p_ret = w.T @ mu
        p_var = w.T @ cov @ w
        return - (p_ret - 0.5 * risk_aversion * p_var)

    # Constraints: Weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

    # Bounds: Long-only (0 to 1) for all assets
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))

    # Initial guess: Evenly distributed
    init_guess = np.ones(n_assets) / n_assets

    result = minimize(
        objective,
        init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'disp': False}
    )

    return result.x if result.success else init_guess

