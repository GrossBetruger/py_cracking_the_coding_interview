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


def rotate_image_right(image: ndarray):
    rotated_image = np.zeros(image.shape)
    for p_vec in np.transpose(np.nonzero(image)):
        set_point(rotate_right(p_vec), rotated_image)
    return rotated_image


def rotate_image_left(image: ndarray):
    rotated_image = np.zeros(image.shape)
    for point_vector in np.transpose(np.nonzero(image)):
        set_point(rotate_left(point_vector), rotated_image)
    return rotated_image


if __name__ == "__main__":
    img = np.identity(100)
    for point in np.transpose(np.nonzero(img)):
        set_point(point, img)
    for point in [(10, 10), (10, 20), (10, 30), (10, 40)]:
        set_point(point, img)

    rot1 = rotate_image_right(img)
    rot2 = rotate_image_right(rot1)
    rot3 = rotate_image_right(rot2)
    rot4 = rotate_image_right(rot3)
    Image.fromarray(img).show()
    Image.fromarray(rot1).show()
    Image.fromarray(rot2).show()
    Image.fromarray(rot3).show()
