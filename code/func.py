import math
import numpy as np
from scipy.special import gammaln


def orthogonal_transformation(A):
    # FOR LOWER TRIANGULAR MATRICES
    n = A.shape[0]
    m = A.shape[1]
    for c in range(min(m, n)):
        # Create the Householder vector
        u = np.zeros(n)
        u[c:] = A[c:, c]
        norm_x = np.linalg.norm(u[c:])
        if norm_x == 0:
            continue
        u[c] += np.sign(u[c]) * norm_x
        norm_u = np.linalg.norm(u)
        if norm_u == 0:
            continue
        u /= norm_u

        # Apply the Householder transformation to A
        for j in range(m):
            A[:, j] -= 2 * np.dot(u, A[:, j]) * u

        # Zero out the elements above the diagonal
        # in column c
        A[:c, c:] = 0

    # Zero out the rest
    A[n-1, c+1:] = 0

    # Return the lower triangular matrix
    return A


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
    sigma = phi
    n = l.shape[0]
    # Initialize the resulting matrix L with zeros
    L = np.zeros((n, n))
    for i in range(n):
        f_i = get_f(i, l, data)
        a = sigma
        # Update sigma using the Euclidean norm
        sigma = np.sqrt(f_i**2 + a**2)
        # Update the diagonal element
        L[i, i] = a / (sigma * phi) * l[i, i]
        for j in range(i):
            g_j = get_g(i, j, l, data)
            # Update the off-diagonal element
            L[i, j] = a / (sigma * phi) * (l[i, j] - f_i * g_j / a**2)
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
    """
    Args:
        Lz: matrix
        mu: dimension of reduced regressor
        
    Returns:
        determinant of L_z,k
    """
    
    det = 1
    for j in range(len(Lz)):
        det *= Lz[j][j]
    return det


def get_det_Lf(Lf):
    """
    Args:
        Lf
        
    Returns:
        determinant of Lf
    """
    det = 1
    for j in range(len(Lf)):
        det *= Lf[j][j]
    return det


def get_det_Vzk(Lzk):
    det_Lzk = get_det_Lz(Lzk)
    det_Vzk = np.power(det_Lzk, -2)
    return det_Vzk


def get_det_Lambda(Lf):
    det_Lf = get_det_Lf(Lf)
    det_Lambda = np.power(det_Lf, -2)
    return det_Lambda


# TODO check the functionality
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
    likelihood += x / 2 * (np.log(det_Vzk) - np.log(det_Vzk_init)) - (delta_t - z + x + 2) / 2 * np.log(det_Lambda) + (delta_0 - z + x + 1) / 2 * np.log(det_Lambda_init)
    
    return likelihood


def reduce_matrix(matrix, mask): # Selects which rows to keep (which regressors to keep) based on the parent vector (here called masking)
    rows_to_keep = np.where(mask)[0]
    reduced_matrix = matrix[rows_to_keep, :]
    return reduced_matrix


def mutate(regressor, p_mut):
    mutated = regressor.copy()
    for i in range(len(mutated)):
        if np.random.rand() < p_mut:
            mutated[i] = 1 - mutated[i] # Flips the bit with probability p_mut
    return mutated 


def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)
    return np.concatenate((parent1[:crossover_point], parent2[crossover_point:])) # Returns an offspring of two parents
    
    
def genetic_algorithm(L_init, L, x, y, t, p_mut=0.1, max_iter=500):
    """
    Algorithm for approximating the optimal structure for regression
    A simple implementation of a stochastic genetic algorithm search for optima for highly unpredictable structure
    Args:
        L_init: initial matrix L
        L: matrix L
        x: dimension of data - nu
        y: dimension of regressor - rho
        t: time - delta t
        p_mut: probability of mutation
        max_iter: maximum number of iterations in this algorithm
    Returns: parent and the highest likelihood value achieved
    """
    
    print(f"Running structure estimation, p_mut = {p_mut}, max_iter = {max_iter}, size of searched space: {2**y}")
    
    # Initialize the matrices and their determinants for the initial state
    # They are calculated here so that they don't have to be recalculated at each iteration
    Lf_init = getL_f(L_init, x)
    det_Lf_init = get_det_Lf(Lf_init)
    Lz_init = getL_z(L_init, x, y)
    det_Lz_init = get_det_Lz(Lz_init)
    
    # These will also play a role in each calculation
    det_Lambda_init = np.power(det_Lf_init, -2)
    det_Vz_init = np.power(det_Lz_init, -2)
    
    # Get the submatrices of L and their determinants
    Lz = getL_z(L, x, y)
    Lf = getL_f(L, x)
    
    det_Vz = get_det_Vzk(Lz)
    det_Lambda = get_det_Lambda(Lf) # This is going to be the same for all model versions
    
    # STEP 1: Initialize parent and its likelihood
    
    parent = np.ones(y) # Initialize the parent, corresponding to the full model
    
    Likelihood_max = get_likelihood(det_Vz_init, det_Lambda_init, det_Vz, det_Lambda, x, y, t) # Set the maximum likelihood as likelihood for the full model
    
    no_improvement_count = 0
    iteration = 0
    
    while iteration < max_iter and no_improvement_count < 3:
        # STEP 2: Itroduce mutations and select the best sub-model
        
        mutated1 = mutate(parent, p_mut) # Mutate the parent (set arbitrary number of mutated versions, basic setting: 2)
        mutated2 = mutate(parent, p_mut)
        
        Lz_mut1 = reduce_matrix(Lz, mutated1)
        det_Vz_mut1 = get_det_Vzk(Lz_mut1 @ Lz_mut1.T)
        Lz_mut1_init = reduce_matrix(Lz_init, mutated1)
        det_Vz_mut1_init = get_det_Vzk(Lz_mut1_init)
        
        Lz_mut2 = reduce_matrix(Lz, mutated2)
        det_Vz_mut2 = get_det_Vzk(Lz_mut2 @ Lz_mut2.T)
        Lz_mut2_init = reduce_matrix(Lz_init, mutated1)
        det_Vz_mut2_init = get_det_Vzk(Lz_mut2_init)
        
        # Set the regressor dimensions for the reduced models
        z1 = np.sum(mutated1)
        z2 = np.sum(mutated2)
        
        # Get respective likelihoods
        Likelihood_mut1 = get_likelihood(det_Vz_mut1_init, det_Lambda_init, det_Vz_mut1, det_Lambda, x, z1, t)
        Likelihood_mut2 = get_likelihood(det_Vz_mut2_init, det_Lambda_init, det_Vz_mut2, det_Lambda, x, z2, t)
        
        # Compare likelihoods
        if Likelihood_mut1 > Likelihood_mut2:
            best_mutation = mutated1
            # best_Likelihood = Likelihood_mut1 # If you want to track these likelihoods too
        else:
            best_mutation = mutated2
            # best_Likelihood = Likelihood_mut2
        
        # STEP 3: Crossover
        
        offspring = crossover(best_mutation, parent)
        
        # Calculate the likelihood for offspring
        Lz_offspring = reduce_matrix(Lz, offspring)
        det_Vz_offspring = get_det_Vzk(Lz_offspring @ Lz_offspring.T)
        Lz_offspring_init = reduce_matrix(Lz_offspring, offspring)
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
            parent = best_mutation # Immediately move to the best mutated version, another option would be to move to the original parent
        
        iteration += 1
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}")
    
    # parent contains the information about which regressors maximize the likelihood
    return parent, Likelihood_max
    
    
    
m = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
mask = np.array([1, 0, 1, 0])  # Keep rows 0 and 2

reduced_m = reduce_matrix(m, mask)
print(reduced_m)

