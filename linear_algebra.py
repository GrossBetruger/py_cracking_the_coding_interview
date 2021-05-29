from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
from numpy import array, ndarray
from numpy.linalg import inv
from scipy.linalg import lu


@dataclass
class Point:
    x: int
    y: int

def gaussian_elimination(coefficients: array, b: array) -> ndarray:
    solutions = inv(coefficients) @ b  # synthetic sugar for: matmul(inv(coefficients), b)
    return solutions


def plu_decomposition(b: ndarray):
    P, L, U = lu(b)
    return P, L, U


def set_point(point: Tuple[int, int], img: ndarray):
    x, y = point
    img[x][y] = 255.


def rotate_right(point_vector: Tuple[int, int]):
    right_rotation_matrix = array(
        [
            [0, 1],
            [-1, 0]
        ]
    )
    return right_rotation_matrix @ point_vector


def rotate_left(point_vector: Tuple[int, int]):
    left_rotation_matrix = array(
        [
            [0, -1],
            [1, 0]
        ]
    )
    return left_rotation_matrix @ point_vector


if __name__ == "__main__":
    point = (10, 10)
    img = np.zeros((100, 100))
    points = [point]
    for p in points:
        set_point(p, img)
    Image.fromarray(img).show()

    # rotate point right ones
    img = np.zeros((100, 100))
    points = [point]
    for p in points:
        x, y = rotate_left(p)
        img[x][y] = 255.
    Image.fromarray(img).show()
