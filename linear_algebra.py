from numpy import array, dot, ndarray
from numpy.linalg import inv
from scipy.linalg import lu


def gaussian_elimination(coefficients: array, b: array) -> ndarray:
    solutions = dot(inv(coefficients), b)
    return solutions


def plu_decomposition(b: ndarray):
    P, L, U = lu(b)
    return P, L, U

