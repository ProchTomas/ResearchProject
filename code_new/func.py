import math
import numpy as np
from scipy.special import gammaln, digamma
from scipy.optimize import minimize
from numpy.linalg import det as det
from scipy.linalg import cholesky, qr, solve_triangular
from scipy.linalg import rq
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar

# HELPER FUNCTIONS

def update_G(G_prev, d, G0, alpha, beta):
    """
    Performs the update V = (alpha+beta)*V_prev + beta*d*d^T + (1-alpha-beta)*V0
    for the decomposition V = G G^T, where G is an upper-triangular matrix
    :arg
        G_prev: The n x n upper-triangular factor of V_prev
        d: The update vector of length n
        G0: The n x n upper-triangular factor of the prior V0
        alpha, beta: Scalar weighting factors

    :returns
        The new upper-triangular factor G
    """
    # 0. Case: no forgetting
    if (alpha and beta) == 0:
        u = d.reshape(-1, 1)
        H = np.hstack([u, G_prev])
        R, _ = rq(H, mode="economic")  # R will be an upper-triangular matrix

        diag = np.diag(R)
        signs = np.where(diag >= 0, 1.0, -1.0)
        G_new = R * signs.reshape(-1, 1)
        return G_new
    elif 1-alpha-beta < 1e-17:
        u = np.sqrt(beta) * d.reshape(-1, 1)
        H = np.hstack([u, np.sqrt(alpha+beta) * G_prev])
        R, _ = rq(H, mode="economic")  # R will be an upper-triangular matrix

        diag = np.diag(R)
        signs = np.where(diag >= 0, 1.0, -1.0)
        G_new = R * signs.reshape(-1, 1)
        return G_new
    else:
        # 1. Scale the components of the update
        g_prev_scaled = np.sqrt(alpha + beta) * G_prev
        u = np.sqrt(beta) * d.reshape(-1, 1)  # Ensure u is a column vector
        g0_scaled = np.sqrt(1 - alpha - beta) * G0

        # 2. Construct a wide matrix H such that V_new = H @ H.T
        H = np.hstack([g_prev_scaled, u, g0_scaled])

        # 3. Perform RQ decomposition: H = R @ Q

        R, _ = rq(H, mode="economic") # R will be an upper-triangular matrix

        # 4. Enforce positive diagonal convention for uniqueness
        signs = np.sign(np.diag(R))
        signs[signs == 0] = 1.0
        G_new = R * signs.reshape(-1, 1)  # Multiply rows of R by signs
    return G_new


def func_F_phi(alpha, beta, gamma, arguments):
    """
    Computes the function F(phi) 
    """
    alpha_0, beta_0, gamma_0, delta_1, delta_0, gx, det_gx, gx_0, det_gx_0, gf, det_gf, gf_0, det_gf_0, V_t_1, V_t, V_0, L_t_1, L_t, L_0, zeta, kappa = arguments

    N = len(gf[0])
    rho = len(gx[0])

    detV_tilde = det(alpha*V_t_1 + beta*V_t + gamma*V_0)
    detL_tilde = det(alpha*L_t_1 + beta*L_t + gamma*L_0)
    delta_tilde = alpha*delta_1 + beta*(delta_1+1) + gamma*delta_0

    f = alpha*(np.log(alpha)-np.log(alpha_0)) + beta*(np.log(beta)-np.log(beta_0)) + gamma*(np.log(gamma)-np.log(gamma_0))
    s = 0
    for j in range(1, N+1):
        s += gammaln((delta_tilde-rho+N+2-j)/2) - alpha*gammaln((delta_1+1-rho+N+2-j)/2) \
             - beta*gammaln((delta_1+1-rho+N+2-j)/2) - gamma*gammaln((delta_0-rho+N+2-j)/2)
    f += s
    f += (alpha+beta)*N*np.log(det_gx) + beta*N/2*np.log(1+zeta) + gamma*N*np.log(det_gx_0) - N/2*np.log(detV_tilde)\
    + ((alpha+beta)*(delta_1-rho+N+1)+beta)*np.log(det_gf) + beta*((delta_1-rho+N+2)/2)*np.log(1+kappa/(1+zeta))\
    + gamma*(delta_0-rho+N+1)*np.log(det_gf_0)-(delta_tilde-rho+N+1)/2*np.log(detL_tilde)
    return f

def map_to_simplex(y):
    # y is length-2 array
    p1 = np.exp(y[0])
    p2 = np.exp(y[1])
    z = 1.0
    denom = p1 + p2 + z
    alpha = p1 / denom
    beta  = p2 / denom
    gamma = z  / denom
    return alpha, beta, gamma

def objective_y(y, args):
    alpha, beta, gamma = map_to_simplex(y)
    return func_F_phi(alpha, beta, gamma, args)

def opt_forget_factors(args):
    y0 = np.array([0.0, 0.0])
    res = minimize(objective_y, y0, args=(args,), method='Nelder-Mead', options={'maxiter': 2000, 'disp': False})
    y_opt = res.x
    alpha, beta, gamma = map_to_simplex(y_opt)
    return alpha, beta, gamma

def func_for_delta0(t_bar, mu, delta0):
    # delta0 must be > 0
    if delta0 <= 0:
        return np.nan  # force solver to avoid non-positive domain
    term1 = 0.5 * np.log(mu)
    term2 = (0.5 * (1 - mu) * t_bar) / (t_bar + (1 - mu) * delta0)
    term3 = np.log(0.5 * delta0) - digamma(0.5 * delta0)
    term4 = -np.log(0.5 * (t_bar + delta0)) + digamma(0.5 * (t_bar + delta0))
    return term1 + term2 + term3 + term4

def find_delta0_integer(t_bar, mu, rho, max_int=1000):
    """
    Find integer delta0 >= 1 that minimizes func_for_delta0.
    """
    candidates = np.arange(rho+1, max_int+1)
    values = [func_for_delta0(t_bar, mu, d) for d in candidates]
    idx = np.nanargmin(np.abs(values))
    return candidates[idx], values[idx]

def opt_prior(y, x, t_bar, N, rho, mu):
    """
    Calculate the optimal prior (V_0, delta_0)
    :arg
        y: response variable
        x: regressors
        t_bar: time to collect V_bar
        N: dim(y)
        rho: dim(x)
        mu: shrinkage parameter
    """
    # V_bar = np.zeros((N+rho, N+rho)) # Begin with zero prior
    # G_bar = V_bar.copy()
    V_0 = np.eye(N+rho)*1e-6 # start with very small V0 for numerical stability
    V_bar = V_0
    G_0 = cholesky(V_bar)
    G_bar = G_0
    a = 0
    b = 0

    for t in range(t_bar):
        d = np.concatenate((y[t], x[t]))
        V_bar += np.outer(d, d)
        G_bar = update_G(G_bar, d, G_0, a, b)

    try:
        delta0_hat = find_delta0_integer(t_bar, mu, rho)
        delta0 = delta0_hat[0]
    except Exception as e:
        print("Solver failed:", e)


    Vyy_bar = V_bar[:N, :N]
    Vyx_bar = V_bar[:N, N:]
    Vxx_bar = V_bar[N:, N:]

    Gyy_bar = G_bar[:N, :N]
    Gyx_bar = G_bar[:N, N:]
    Gxx_bar = G_bar[N:, N:]

    Vyy0 = (mu*delta0/(t_bar+(1-mu)*delta0)*Vyy_bar +
            mu*t_bar/(t_bar*(1-mu)+(1-mu)**2*delta0)*Vyx_bar@np.linalg.inv(Vxx_bar)@Vyx_bar.T)
    Vyx0 = mu/(1-mu)*Vyx_bar
    Vxx0 = mu/(1-mu)*Vxx_bar


    Gyy0 = np.sqrt(mu*delta0/(t_bar+(1-mu)*delta0))*Gyy_bar
    Gyx0 = np.sqrt(mu/(1-mu))*Gyx_bar
    Gxx0 = np.sqrt(mu/(1-mu))*Gxx_bar

    V_0 = np.block([[Vyy0, Vyx0], [Vyx0.T, Vxx0]])
    G_0 = np.block([[Gyy0, Gyx0], [np.zeros_like(Gyx0.T), Gxx0]])

    return V_0, G_0, delta0

def get_likelihood(delta, t, rho_m, N, mu,
                   det_Lambda_log, det_Lambda0_log,
                   det_Vxx_log, det_Vxx0_log):
    """
    log-likelihood function for the GA
    """
    if rho_m == 0:
        return -np.inf
    gammaterm = 0.0
    for j in range(N):
        gammaterm += gammaln((delta + t - rho_m + N + 1 - j) / 2.0) - gammaln((delta - rho_m + N + 1 - j) / 2.0)

    ll = gammaterm - (N/2.0) * det_Vxx_log - (delta - rho_m + t + N)/2.0 * det_Lambda_log + (N/2.0) * det_Vxx0_log + (delta - rho_m + N)/2.0 * det_Lambda0_log

    return ll

def reduce_matrix(matrix, row_mask=None, col_mask=None):
    """
    Reduce matrix by keeping only rows/columns where mask == 1
    :arg
        matrix : Input matrix to be reduced
        row_mask : np.ndarray of 0s/1s (optional)
        col_mask : np.ndarray of 0s/1s (optional)
    :returns
        np.ndarray - reduced matrix
    """
    reduced_matrix = matrix

    if row_mask is not None:
        rows_to_keep = np.where(row_mask)[0]
        reduced_matrix = reduced_matrix[rows_to_keep, :]

    if col_mask is not None:
        cols_to_keep = np.where(col_mask)[0]
        reduced_matrix = reduced_matrix[:, cols_to_keep]

    return reduced_matrix

def mutate(regressor, p_mut):
    mutated = regressor.copy()
    for i in range(len(mutated)):
        if np.random.rand() < p_mut:
            mutated[i] = 1 - int(mutated[i])
    # ensure at least one included
    if not np.any(mutated):
        idx = np.random.randint(len(mutated))
        mutated[idx] = 1
    return mutated

def crossover(parent1, parent2):
    if len(parent1) < 2:
        return parent1.copy()
    cp = np.random.randint(1, len(parent1))
    return np.concatenate((parent1[:cp], parent2[cp:]))

def genetic_algorithm(Vyy, Vyy0, Vyx, Vyx0, Vxx, Vxx0, time_idx, d0, mu,
                      batch_size, p_mut, max_iter, decay_rate):
    """Performs the genetic algorithm on regressors"""
    
    Lambda = Lambda0 = Vyy0 - Vyx0@np.linalg.inv(Vxx0)@Vyx0.T
    n = Lambda.shape[0]  # dimension of augmented vector [y;x] is n? (here N?)
    m = Vxx.shape[0]
    print(f"GA: searching 2^{m} space. start p_mut={p_mut}, max_iter={max_iter}")

    # precompute logdets for Lambda and Lambda0
    logdet_Lambda = np.log(det(Lambda))
    logdet_Lambda0 = np.log(det(Lambda0))

    # initial parent = keep all
    parent = np.zeros(m, dtype=int)
    rho_ = np.sum(parent)

    # compute initial Vxx / Vxx0 logdets (full model)
    logdet_Vxx_full = np.log(det(Vxx))
    logdet_Vxx0_full = np.log(det(Vxx0))

    likelihood_max = get_likelihood(d0, time_idx, rho_, n, mu,
                                    logdet_Lambda, logdet_Lambda0,
                                    logdet_Vxx_full, logdet_Vxx0_full)
    # un-comment for plots
    # ll_history = []
    # pop_history = []

    iteration = 0
    while iteration < int(max_iter):
        mutations = [mutate(parent, p_mut) for _ in range(batch_size)]
        liks = []

        # un-comment for plots
        # likelihoods = []

        for mutated in mutations:
            rho_mut = int(np.sum(mutated))

            Vxx_m = reduce_matrix(Vxx, row_mask=mutated, col_mask=mutated)
            Vxx0_m = reduce_matrix(Vxx0, row_mask=mutated, col_mask=mutated)
            Vyx_m = reduce_matrix(Vyx, row_mask=None, col_mask=mutated)
            Vyx0_m = reduce_matrix(Vyx0, row_mask=None, col_mask=mutated)

            ld_Vxx_m = np.log(det(Vxx_m))
            ld_Vxx0_m = np.log(det(Vxx0_m))
            ld_Lambda_m = np.log(det(Vyy - Vyx_m@np.linalg.inv(Vxx_m)@Vyx_m.T))
            ld_Lambda0_m = np.log(det(Vyy - Vyx0_m@np.linalg.inv(Vxx0_m)@Vyx0_m.T))

            lk = get_likelihood(d0, time_idx, rho_mut, n, mu, ld_Lambda_m, ld_Lambda0_m, ld_Vxx_m, ld_Vxx0_m)
            # un-comment for plots
            # likelihoods.append(lk)

            liks.append((lk, mutated))

        best_mutation = max(liks, key=lambda x: x[0])[1].astype(int)

        # evaluate best_mutation
        rho_b = int(np.sum(best_mutation))

        Vxx_b = reduce_matrix(Vxx, row_mask=best_mutation, col_mask=best_mutation)
        Vxx0_b = reduce_matrix(Vxx0, row_mask=best_mutation, col_mask=best_mutation)
        Vyx_b = reduce_matrix(Vyx, row_mask=None, col_mask=best_mutation)
        Vyx0_b = reduce_matrix(Vyx0, row_mask=None, col_mask=best_mutation)

        ld_Vxx_b = np.log(det(Vxx_b))
        ld_Vxx0_b = np.log(det(Vxx0_b))
        ld_Lambda_b = np.log(det(Vyy - Vyx_b@np.linalg.inv(Vxx_b)@Vyx_b.T))
        ld_Lambda0_b = np.log(det(Vyy - Vyx0_b@np.linalg.inv(Vxx0_b)@Vyx0_b.T))

        lk_b = get_likelihood(d0, time_idx, rho_b, n, mu, ld_Lambda_b, ld_Lambda0_b, ld_Vxx_b, ld_Vxx0_b)

        if lk_b > likelihood_max:
            parent = best_mutation.copy()
            likelihood_max = lk_b
            # print(f"New best (mut): {parent}, lik={likelihood_max:.4f}")

        # crossover
        offspring = crossover(parent, best_mutation).astype(int)

        rho_off = int(np.sum(offspring))

        Vxx_off = reduce_matrix(Vxx, row_mask=offspring, col_mask=offspring)
        Vxx0_off = reduce_matrix(Vxx0, row_mask=offspring, col_mask=offspring)
        Vyx_off = reduce_matrix(Vyx, row_mask=None, col_mask=offspring)
        Vyx0_off = reduce_matrix(Vyx0, row_mask=None, col_mask=offspring)

        ld_Vxx_off = np.log(det(Vxx_off))
        ld_Vxx0_off = np.log(det(Vxx0_off))
        ld_Lambda_off = np.log(det(Vyy - Vyx_off@np.linalg.inv(Vxx_off)@Vyx_off.T))
        ld_Lambda0_off = np.log(det(Vyy - Vyx0_off@np.linalg.inv(Vxx0_off)@Vyx0_off.T))

        lk_off = get_likelihood(d0, time_idx, rho_off, n, mu, ld_Lambda_off, ld_Lambda0_off, ld_Vxx_off, ld_Vxx0_off)

        if lk_off > likelihood_max:
            parent = offspring.copy()
            likelihood_max = lk_off
            # print(f"New best (off): {parent}, lik={likelihood_max:.4f}")

        # un-comment for plots
        # if iteration % 5 == 0:
        #     pop_history.append(np.array(mutations))
        #     ll_history.append(np.array(likelihoods))
        
        # update mutation rate + iter
        p_mut *= decay_rate
        iteration += 1

    # OPTIONALLY: plot the results

    # n_iters = len(ll_history)
    # pop_size = len(ll_history[0])
    #
    # flat_pop = np.vstack(pop_history)
    # xticks = np.arange(n_iters) * 5
    # xtick_positions = np.arange(pop_size // 2, pop_size * n_iters, pop_size)
    # ll_array = np.array(ll_history)
    #
    # # --- Plot setup ---
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False, gridspec_kw={'height_ratios': [1, 1.5]})
    #
    # # --- Top: Likelihoods ---
    # for i in range(n_iters):
    #     ax1.scatter([i * 5] * pop_size, ll_array[i], color='tab:blue', alpha=0.7)
    # plt.gca().set_facecolor("white")
    # ax1.set_ylabel("log-likelihood")
    # ax1.set_title("Population Likelihoods")
    # ax1.grid(True)
    #
    # # --- Bottom: Binary Population Heatmap ---
    # # Transpose to shape (features, time points)
    # ax2.imshow(flat_pop.T, aspect='auto', cmap='Greys', interpolation='nearest')
    # ax2.set_ylabel("Regressor Index")
    # ax2.set_xlabel("Generation")
    # ax2.set_title("Population Feature Selection")
    #
    # ax2.set_xticks(xtick_positions)
    # ax2.set_xticklabels(xticks)
    #
    # plt.tight_layout()
    # plt.show()

    return parent, likelihood_max


def objective_omega(omega, d, sigma, a_prev):
    # e = d + omega ** 2 * sigma
    # x = d - d@np.linalg.inv(e)@d
    # return np.log(det(e)) + np.dot(a_prev, x@a_prev)
    e_inv = np.linalg.inv(d + omega ** 2 * sigma)
    return omega*np.trace(e_inv @ sigma) + omega* a_prev@d@e_inv@sigma@e_inv@d@a_prev

def get_omega(d, sigma, a_prev,
                      omega_min=1e-8, omega_max=1e3):
    """
    Minimizes the objective function directly.
    """
    res = minimize_scalar(
        objective_omega,
        args=(d, sigma, a_prev),
        bounds=(omega_min, omega_max),
        method='bounded'
    )
    if not res.success:
        raise RuntimeError(f"Minimization failed: {res.message}")
    return res.x

# def lprime_chol(omega, d, sigma, y_mu, a_prev):
#     e = d + 0.5 * omega**2 * sigma
#     L, lower = cho_factor(e)
#     b = 2.0 * d.dot(a_prev) + omega * y_mu
#
#     x = cho_solve((L, lower), b)
#     Y = cho_solve((L, lower), sigma)
#
#     trace_term = np.trace(Y)
#     quad_term = x.dot(sigma.dot(x))
#
#     return 0.5 * omega * trace_term - y_mu.dot(x) + 0.5 * omega * quad_term
#
#
# def get_omega(D, sigma, y_mu, a_prev):
#     sol = root_scalar(
#         lprime_chol,
#         args=(D, sigma, y_mu, a_prev),
#         bracket=[1e-6, 10.0],
#         method='brentq'
#     )
#     return sol.root

def sample_matrix_normal(p_hat, s, g_x, n=1, rng=None):
    """Sample matrix P from its posterior
    :arg
      p_hat : (N, rho) mean matrix
      s     : (N, N) positive-definite row-covariance (Sigma)
      V     : (rho, rho) positive-definite matrix appearing as in your formula (we use V so Sigma_col = V^{-1})
      n     : number of samples to draw
      rng   : np.random.Generator or None

    :return
      samples : (n, r, c) if n>1 else (r, c)
    """
    if rng is None:
        rng = np.random.default_rng()

    p_hat = np.asarray(p_hat)
    s = np.asarray(s)
    v_x = g_x@g_x.T
    v_x = np.asarray(v_x)
    r, c = p_hat.shape
    assert s.shape == (r, r)
    assert v_x.shape == (c, c)

    # Cholesky of S (lower triangular): S = L_s L_s^T
    s_inv = np.linalg.inv(s)
    L_s = cholesky(s_inv, lower=True)

    # Use Cholesky of V and triangular solve to get a factor of V^{-1} without explicit inverse:
    # V = L_v L_v^T  =>  V^{-1} = (L_v^{-1}) (L_v^{-1})^T
    L_v = cholesky(v_x, lower=True)
    # L_v_inv is lower triangular with L_v_inv @ L_v_inv.T = V^{-1}
    L_v_inv = solve_triangular(L_v, np.eye(c), lower=True)

    # draw standard normals Z ~ N(0, I) of shape (r, c) and transform
    if n == 1:
        z = rng.standard_normal(size=(r, c))
        p = p_hat + L_s @ z @ L_v_inv.T   # L_v_inv.T is the matching right factor
        return p
    else:
        samples = np.empty((n, r, c))
        for i in range(n):
            z = rng.standard_normal(size=(r, c))
            samples[i] = p_hat + L_s @ z @ L_v_inv.T
        return samples


def inv_and_logdet_spd(A, eps=1e-8):
    A = 0.5 * (A + A.T)  # enforce symmetry
    w, V = np.linalg.eigh(A)
    w_clipped = np.clip(w, eps, None)

    A_inv = (V / w_clipped) @ V.T
    logdet = np.sum(np.log(w_clipped))
    return A_inv, logdet



def func_F_for_H(phi, phi_0, h_new, h_prev):
    if not (0.0 < phi < 1.0):
        return np.inf

    try:
        h_prev_inv, logdet_p = inv_and_logdet_spd(h_prev)
        h_new_inv,  logdet_n = inv_and_logdet_spd(h_new)
    except np.linalg.LinAlgError:
        return np.inf

    h_tilde_inv = phi * h_prev_inv + (1 - phi) * h_new_inv

    # logdet of h_tilde_inv
    _, logdet_t = inv_and_logdet_spd(np.linalg.inv(h_tilde_inv))

    return (
        phi * np.log(phi / phi_0)
        + (1 - phi) * np.log((1 - phi) / (1 - phi_0))
        - 0.5 * logdet_t
        + 0.5 * phi * logdet_p
        + 0.5 * (1 - phi) * logdet_n
    )



def optimize_H(phi_0, h_prev, h_new):
    res = minimize_scalar(
        func_F_for_H,
        bounds=(1e-6, 1 - 1e-6),
        args=(phi_0, h_new, h_prev),
        method="bounded"
    )

    phi_opt = res.x
    return phi_opt * h_prev + (1 - phi_opt) * h_new


