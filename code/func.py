import math
import numpy as np
from scipy.special import gammaln


def orthogonal_transformation(A):
    # FOR LOWER TRIANGULAR MATRICES
    n = A.shape[0]
    m = A.shape[1]
    for c in range(min(n, m)):
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
        A[:c, c] = 0
    # Return the lower triangular matrix
    return np.transpose(A)


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


def get_det_Lzk(Lz, E, mu):
    """
    Args:
        Lz: matrix
        E: permutation matrix
        mu: dimension of reduced regressor
        
    Returns:
        determinant of L_z,k
    """
    Lzk = orthogonal_transformation(E @ np.linalg.inv(Lz)) # apply the householder transformation
    
    det = 1
    for j in range(mu):
        det *= Lzk[j][j]
    return 1 / det # det L_z,k = 1 / det L_z,k^-1


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

# TODO check the functionality
def get_likelihood(L_init, L, E, x, y, z, t):
    """
    Args:
        L_init: initial matrix L
        L: matrix L
        E: permutation matrix
        x: dimension of data - nu
        y: dimension of regressor - rho
        z: dimension of reduced regressor - mu
        t: time - delta t
        
    Returns:
        likelihood value
    """
    Lz = getL_z(L, x, y)
    Lf = getL_f(L, x)
    
    Lz_init = getL_z(L_init, x, y)
    Lf_init = getL_f(L_init, x)
    
    det_Lzk_init = get_det_Lzk(Lz_init, E, z)
    det_Lf_init = get_det_Lf(Lf_init)
    
    det_Lzk = get_det_Lzk(Lz, E, z)
    det_Lf = get_det_Lf(Lf)
    
    delta_0 = 10 + y
    delta_t = delta_0 + t
    
    likelihood = 0
    
    for j in range(x):
        likelihood += gammaln((delta_t - z + x + 2 - j) / 2) - gammaln((delta_0 - z + x + 1 - j) / 2)
    likelihood += x * (np.log(det_Lzk) - np.log(det_Lzk_init)) + (delta_t - z + x + 2) * np.log(det_Lf) - (delta_0 - z + x + 1) * np.log(det_Lf_init)
    
    return likelihood


# TODO structure estimation algorithm

def generate_permutations(matrix, index, result): # generator of matrices E_k
    if index == len(matrix):
        if len(matrix) > 0: # excludes the empty matrix
            result.append(matrix.copy())
        return

    generate_permutations(matrix, index + 1, result)

    matrix = np.delete(matrix, index, axis=0)
    generate_permutations(matrix, index, result)


def generate_identity_permutations(n):
    identity_matrix = np.eye(n)
    permutations = []
    generate_permutations(identity_matrix, 0, permutations)
    return permutations