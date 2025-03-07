import math
import numpy as np
from scipy.special import gammaln
import func


def householder_transform(input_matrix):
    """Orthogonal transformation into lower triangular matrix using Householder reflections.
    Args:
        input_matrix: matrix to transform
    Returns:
        lower triangular matrix
    """
    A = input_matrix.copy()
    n, m = A.shape

    for c in range(min(m, n)):
        # Create the Householder vector
        u = np.zeros(n)
        u[c:] = A[c:, c]
        norm_x = np.linalg.norm(u[c:])
        if norm_x == 0:
            continue
        sign = np.sign(u[c]) if u[c] != 0 else 1 # Ensure consistency of sign
        u[c] += sign * norm_x
        u /= np.linalg.norm(u)

        # Apply the Householder transformation (vectorized)
        A -= 2 * np.outer(u, u @ A)
    # Zeros out elements above the diagonal
    return np.tril(A)


# Intermediate functions f and g
def get_f(i, l, data):
    # i: The current row index
    return np.dot(l[i, :i+1], data[:i+1])


def get_g(i, j, l, data):
    # i: The current row index
    # j: The current column index
    return np.sum(l[j:i, j] * [get_f(k, l, data) for k in range(j, i)])


def refill(l, data, phi):
    """ update matrix L

    Args:
        l: previous L
        data: current data point
        phi: exponential forgetting parameter

    Returns:
        updated L
    """
    sigma = np.sqrt(phi)
    n = l.shape[0]
    # Initialize the resulting matrix L with zeros
    L = np.zeros((n, n))
    for i in range(n):
        f_i = get_f(i, l, data)
        a = sigma / np.sqrt(phi)
        # Update sigma using the Euclidean norm
        sigma = np.sqrt(f_i**2 + a**2)
        # Update the diagonal element
        L[i, i] = a / sigma * l[i, i]
        for j in range(i):
            g_j = get_g(i, j, l, data)
            # Update the off-diagonal element
            L[i, j] = a / sigma * (l[i, j] - f_i * g_j / a**2)
    return L


def getL_z(matrix, x, y): # returns L_z
    """
    Args:
        matrix: L
        x: dimension of data - nu
        y: dimension of regressor - rho

    Returns:
        matrix Lz
    """
    return matrix[x:x+y, x:x+y]


def getL_f(matrix, x):
    """
    Args:
        matrix: L
        x: dimension of data - nu
        
    Returns:
        matrix Lf
    """
    return matrix[:x, :x]


def getL_zf(matrix, x, y):
    """
    Args:
        matrix: L
        x: dimension of data - nu
        y: dimension of regressor - rho
        
    Returns:
        matrix L_zf
    """
    return matrix[x:x+y, :x]


def update_V(v, data):
    return v + np.outer(data, data)


def get_det_Lz(Lz):
    """Computes the determinant as a product of diagonal elements
    This is done to avoid matrix multiplication in computing determinant of Vzk
    Args:
        Lz: lower triangular matrix
    Returns:
        determinant of L_z,k
    """
    
    det = 1
    for j in range(len(Lz)):
        det *= Lz[j][j]
    return det


def get_det_Lf(Lf):
    """Computes the determinant as a product of diagonal elements
    This is done to avoid matrix multiplication in computing determinant of Lambda
    Args:
        Lf: lower triangular matrix
    Returns:
        determinant of L_f
    """
    det = 1
    for j in range(len(Lf)):
        det *= Lf[j][j]
    return det


def get_det_Vzk(Lzk):
    """Takes in Lzk and computes the determinant of Vzk within this function
    Args:
        Lzk: lower triangular matrix
    Retruns:
        determinant of submatrix Vzk
    """
    det_Lzk = get_det_Lz(Lzk)
    # Handle small determinants to override the division by zero issue (does not happen too often)
    epsilon = 1e-12
    safe_det_Lzk = np.maximum(det_Lzk, epsilon)
    
    det_Vzk = np.power(safe_det_Lzk, -2)
    return det_Vzk


def get_det_Lambda(Lf):
    """Takes in Lzk and computes the determinant of Lambda within this function
    Args:
        Lf: lower triangular matrix
    Returns:
        determinant of Lambda
    """
    det_Lf = get_det_Lf(Lf)
    det_Lambda = np.power(det_Lf, -2)
    return det_Lambda


def get_likelihood(det_Vzk_init, det_Lambda_init, det_Vzk, det_Lambda, x, z, t):
    """
    Args:
        det_Vzk_init: determinant of initial Vzk
        det_Lambda_init: determinant of initial Lambda
        det_Vzk: determinant of current Vzk (|Vzk| = |Lzk|^-2)
        det_Lambda: determinant of current Lambda (|Lambda| = |Lf|^-2)
        x: dimension of data - nu
        z: dimension of reduced regressor - mu (= y, for the full model)
        t: time
    Returns:
        likelihood value
    """
    
    delta_0 = 10 + z
    delta_t = delta_0 + t
    
    likelihood = 0
    
    # It is important to count with the initial determinants and det_Lambda, though they remain the same. This is because of the different scaling at each step
    # This could perhaps be optimized by precomputing the differences for det_Lambda
    for j in range(x):
        likelihood += gammaln((delta_t - z + x + 2 - j) / 2) - gammaln((delta_0 - z + x + 1 - j) / 2)
    likelihood -= x / 2 * (np.log(det_Vzk) - np.log(det_Vzk_init)) - (delta_t - z + x + 2) / 2 * np.log(det_Lambda) + (delta_0 - z + x + 1) / 2 * np.log(det_Lambda_init)
    
    return likelihood


def reduce_matrix(matrix, mask):
    rows_to_keep = np.where(mask)[0] # Selects which rows to keep (which regressors to keep) based on the parent vector (here called mask)
    reduced_matrix = matrix[rows_to_keep, :]
    return reduced_matrix


def mutate(regressor, p_mut):
    mutated = regressor.copy()
    for i in range(len(mutated)):
        if np.random.rand() < p_mut:
            mutated[i] = 1 - mutated[i] # Flips the bit with probability p_mut
            
    # Ensure at least one regressor is included
    if not np.any(mutated):
        idx = np.random.randint(len(mutated))
        mutated[idx] = 1  # Randomly activate one regressor
        
    return mutated 


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])) # Returns an offspring of two parents


def softmax_selection(likelihoods):
    # Shift likelihoods to prevent infinities after exponential is applied
    likelihoods -= np.max(likelihoods)
    exp_likelihoods = np.exp(likelihoods)
    probabilities = exp_likelihoods / np.sum(exp_likelihoods)
    return np.random.choice(len(likelihoods), p=probabilities)


def genetic_algorithm(L_init, L, x, y, t, batch_size, p_mut=0.1, max_iter=2, decay_rate=0.995):
    """
    Algorithm for approximating the optimal structure for regression
    A simple implementation of a stochastic genetic algorithm search for optima for highly unpredictable structure
    Args:
        L_init: initial matrix L
        L: matrix L
        x: dimension of data - nu
        y: dimension of regressor - rho
        t: time - delta t
        batch_size: number of mutated version generated each iteration
        p_mut: probability of mutation
        max_iter: maximum number of iterations in this algorithm
        decay_rate: rate of exponential decay for p_mut
    Returns:
        tuple of (parent, the highest likelihood value achieved)
    """
    
    print(f"Running structure estimation, p_mut = {p_mut}, max_iter = {max_iter}, size of searched space: {2**y}")
    
    # STEP 0: Initialize the matrices and their determinants for the initial state, possibly save and load these for very large matrices
    Lf_init = getL_f(L_init, x)
    Lz_init = getL_z(L_init, x, y)
    det_Lambda_init = get_det_Lambda(Lf_init)
    det_Vz_init = get_det_Vzk(Lz_init)
    
    # Get the submatrices of current matrix L and their determinants
    Lz = getL_z(L, x, y)
    Lf = getL_f(L, x)
    
    det_Vz = get_det_Vzk(Lz)
    det_Lambda = get_det_Lambda(Lf) # This is going to be the same for all model versions
    
    # STEP 1: Initialize parent and its likelihood
    
    parent = np.ones(y) # Initialize the parent, corresponding to the full model
    
    Likelihood_max = get_likelihood(det_Vz_init, det_Lambda_init, det_Vz, det_Lambda, x, y, t) # Set the maximum likelihood as likelihood for the full model
    
    no_improvement_count = 0
    iteration = 0
    
    while iteration < max_iter and no_improvement_count < 10:
        # STEP 2: Itroduce mutations and select the best sub-model
        mutations = [mutate(parent, p_mut) for _ in range(batch_size)]
        likelihoods = []
        
        for mutated in mutations:
            Lz_mut = reduce_matrix(Lz, mutated)
            Lz_mut_init = reduce_matrix(Lz_init, mutated)
            
            Lz_mut = func.householder_transform(Lz_mut)
            Lz_mut_init = func.householder_transform(Lz_mut_init)
            
            det_Vz_mut = get_det_Vzk(Lz_mut)
            det_Vz_mut_init = get_det_Vzk(Lz_mut_init)
            
            z_mut = np.sum(mutated)
            likelihood = get_likelihood(det_Vz_mut_init, det_Lambda_init, det_Vz_mut, det_Lambda, x, z_mut, t)
            likelihoods.append((likelihood, mutated))
            
            # For soft-max selection
            # likelihoods.append(likelihood)
        
        best_mutation = max(likelihoods, key=lambda x: x[0])[1]
        
        # For soft-max selection
        # selected_idx = softmax_selection(np.array(likelihoods))
        # best_mutation = mutations[selected_idx]

        # STEP 3: Crossover
        offspring = crossover(best_mutation, parent)

        Lz_offspring = reduce_matrix(Lz, offspring)
        Lz_offspring_init = reduce_matrix(Lz_init, offspring)

        Lz_offspring = func.householder_transform(Lz_offspring)
        Lz_offspring_init = func.householder_transform(Lz_offspring_init)

        det_Vz_offspring = get_det_Vzk(Lz_offspring)
        det_Vz_offspring_init = get_det_Vzk(Lz_offspring_init)
        
        z_offspring = np.sum(offspring)
        
        Likelihood_offspring = get_likelihood(det_Vz_offspring_init, det_Lambda_init, det_Vz_offspring, det_Lambda, x, z_offspring, t)
        
        # STEP 4: Evaluate offspring
        if Likelihood_offspring > Likelihood_max:
            parent = offspring
            Likelihood_max = Likelihood_offspring
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            parent = best_mutation
        
        p_mut *= decay_rate
        iteration += 1
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Max Likelihood: {Likelihood_max}")
    
    # parent contains the information about which regressors maximize the likelihood
    return parent, Likelihood_max


def F_phi(p_init, p, m, N):
    det_m = 0
    for i in range(len(m[0])):
        det_m += np.log(np.power(m[i][i], 2))
    f_value = p * np.log(p / p_init) + (1 - p) * np.log((1 - p) / (1 - p_init)) + N/2 * det_m
    return f_value


def M_phi(L_z, z, p):
    m_square_root = np.block([[np.sqrt(p) * z.T], [np.linalg.inv(L_z)]])
    m = householder_transform(m_square_root)
    return m


def opt_forgetting_factor(L_z, z, N, phi_init, a=0, b=1, tol=1e-6):
    """
    Golden section search algorithm to find the optimal forgetting factor.
    Args:
        L_z: submatrix of L
        z: current regressor
        N: dimension of data - nu
        phi_init: initial guess for the optimal forgetting factor
    Returns:
        optimal forgetting factor at for the contemporary data
    """
    gr = (1 + 5 ** 0.5) / 2

    # Calculate initial test points
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    # Evaluate function at test points
    fc = F_phi(phi_init, c, M_phi(L_z, z, c), N)
    fd = F_phi(phi_init, d, M_phi(L_z, z, d), N)

    while abs(b - a) > tol:
        if fc < fd:
            # Minimum is in [a, d]
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = F_phi(phi_init, c, M_phi(L_z, z, c), N)
        else:
            # Minimum is in [c, b]
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = F_phi(phi_init, d, M_phi(L_z, z, d), N)

    # Return midpoint of final interval
    x = (a + b) / 2
    return x
