import math
import numpy as np


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
    return matrix[x:x+y, x:x+y]


def getL_f(matrix, x): # returns L_f
    return matrix[:x, :x]


def getL_zf(matrix, x, y): # returns L_zf
    return matrix[x:x+y, :x]


def update_V(v, data):
    return v + np.outer(data, data)


def get_det_Vzk(K): # returns determinant of V_z,k^-1 as product of diagonal elements of K = L_z,k^-1
    s = 1
    for j in range(len(K)):
        s *= K[j][j]
    return np.power(s, 2)


def get_det_Lambda(Lf): # returns determinant of Lambda^-1
    d = 1
    for j in range(len(Lf)):
        d *= Lf[j][j]
    return np.power(d, -2)