from numpy import array, dot, ndarray
from numpy.linalg import inv


def gaussian_elimination(coefficients: array, b: array) -> ndarray:
    solutions = dot(inv(coefficients), b)
    return solutions
