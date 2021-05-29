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


def apply_shear(point_vector: Tuple[int, int]):
    shear = array(
        [
            [1, 1],
            [0, 1]
        ]
    )
    return shear @ point_vector


def apply_shear_to_matrix(image: ndarray):
    sheared_image = np.zeros(image.shape)
    for p_vec in np.transpose(np.nonzero(image)):
        set_point(apply_shear(p_vec), sheared_image)
    return sheared_image


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
    # Rotation transformations
    img = np.identity(200)
    for point in np.transpose(np.nonzero(img)):
        set_point(point, img)
    for point in [(10, 10), (10, 20), (10, 30), (10, 40)]:
        set_point(point, img)

    rot1 = rotate_image_right(img)
    rot2 = rotate_image_right(rot1)
    rot3 = rotate_image_right(rot2)
    rot4 = rotate_image_right(rot3)
    upper_side = np.hstack((img, rot1))
    lower_side = np.hstack((rot3, rot2))
    Image.fromarray(np.vstack((upper_side, lower_side))).show()

    # Shear transformations
    img = np.zeros((200, 200))
    for point in [(10, 10), (10, 20), (10, 30)]:
        set_point(point, img)
    shear1 = apply_shear_to_matrix(img)
    shear2 = apply_shear_to_matrix(shear1)
    shear3 = apply_shear_to_matrix(shear2)
    shear4 = apply_shear_to_matrix(shear3)
    shear5 = apply_shear_to_matrix(shear4)
    shear6 = apply_shear_to_matrix(shear5)
    upper_side = np.hstack((shear1, shear2, shear3))
    lower_side = np.hstack((shear4, shear5, shear6))
    Image.fromarray(np.vstack((upper_side, lower_side))).show()
