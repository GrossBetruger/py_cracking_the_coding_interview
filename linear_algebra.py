from dataclasses import dataclass
from enum import Enum, auto
from math import radians, sin, cos
from typing import Tuple, Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import array, ndarray
from numpy.linalg import inv
from scipy.linalg import lu
from sklearn.linear_model import LinearRegression
from sklearn.datasets._samples_generator import make_blobs
from sklearn.svm import SVC


@dataclass
class Point:
    x: int
    y: int


class LeastSquares:
    def __init__(self):
        self.pred = None

    def train(self,  x: ndarray, y: ndarray):
        try:
            assert len(x) == len(y)
        except AssertionError:
            raise ValueError(f"lengths x, y differ: {len(x), len(y)}")

        n = len(x)
        m = (n * (x * y).sum() - x.sum() * y.sum()) / (n * (x ** 2).sum() - x.sum() ** 2)
        b = (y.sum() - m * x.sum()) / n
        self.pred = lambda _x: m * _x + b

    def predict(self, x):
        if self.pred is None:
            raise Exception("model not trained, run `train` method first")
        return self.pred(x)


class Orientation(Enum):
    XPositiveYPositive = auto()
    XPositiveYNegative = auto()
    XNegativeYPositive = auto()
    XNegativeYNegative = auto()
    Absolute = auto()


Degree = Annotated[float, (0, 360)]


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


def two_d_vector_from_magnitude_and_angle(magnitude: float,
                                          angle: Degree,
                                          orientation: Optional[Orientation] = Orientation.Absolute) -> ndarray:
    radian_angle = radians(angle)
    if orientation is Orientation.XPositiveYPositive:
        v = array(
            [magnitude * sin(radian_angle),  # X component
             magnitude * cos(radian_angle)]  # Y Component
        )

    elif orientation is Orientation.XPositiveYNegative:
        v = array(
            [magnitude * cos(radian_angle),  # X component
             magnitude * sin(radian_angle) * -1]  # Y Component
        )

    elif orientation is Orientation.XNegativeYPositive:
        v = array(
            [magnitude * sin(radian_angle) * -1,  # X component
             magnitude * cos(radian_angle) ]  # Y Component
        )

    elif orientation is Orientation.XNegativeYNegative:
        v = array(
            [magnitude * cos(radian_angle) * -1,  # X component
             magnitude * sin(radian_angle) * -1]  # Y Component
        )

    elif orientation is Orientation.Absolute:
        v = array(
            [magnitude * cos(radian_angle),  # X component
             magnitude * sin(radian_angle)]  # Y Component
        )

    else:
        raise ValueError(f"Invalid Orientation: {orientation}")

    return v


def multiply_matrices(mat1: ndarray, mat2: ndarray) -> ndarray:
    assert mat1.shape[1] == mat2.shape[0]  # "inside" dimensions must be equal for operation to be defined
    product_matrix = np.zeros((mat1.shape[0], mat2.shape[1]))  # "outside" dimensions create dimensionality of new mat

    for i, row in enumerate(mat1):
        new_row = []
        for column in mat2.transpose():
            new_cell = np.dot(row, column)
            new_row.append(new_cell)
        product_matrix[i] = np.array(new_row)

    return product_matrix


def take_derivative(function_parameters: ndarray) -> ndarray:
    """
    :param function_parameters: numpy array of coefficients
    :return: derivative as numpy array of coefficients
    """

    degree = len(function_parameters)
    d: ndarray = np.eye(degree, k=1)
    derivative_matrix = np.zeros(shape=(degree, degree))
    for i, row in enumerate(d):
        derivative_matrix[i] = np.where(row > 0, i+1, 0)

    #  example derivative matrix (degree = 4; which mathematically is degree 3)
    # [[0. 1. 0. 0.]
    #  [0. 0. 2. 0.]
    #  [0. 0. 0. 3.]
    #  [0. 0. 0. 0.]]

    return derivative_matrix @ function_parameters


def linear_regression(A: ndarray, y: ndarray) -> ndarray:
    """
    :param A: approximated system of linear equations as matrix of coefficients:
    :param y: observed y values
    :return: solution of system as a column of the shape (slope, y-intercept)
    """
    return inv(A.transpose() @ A) @ A.transpose() @ y # (A^(T) A)^(-1) A^(T)Y


def linear_regression_demo():
    x = np.random.uniform(-7, 7, (50, 1))
    y = array([elem*3 + 7 + np.random.uniform(-5, 5)
               for elem in x])
    reg = LinearRegression().fit(x, y)
    y_hat = reg.predict(x)
    # assert np.allclose(y_hat, y, atol=2)

    plt.scatter(x, y)
    plt.plot(x, y_hat, color="red")
    plt.show()


def svm_demo():
    X, y = make_blobs(n_samples=150, centers=2, random_state=42)

    clf = SVC(kernel='linear')
    clf.fit(X, y)

    clf.predict(X)
    print(y)
    # plt.scatter(X, y)
    # plt.show()
    # plt.contour(clf, X[:, 0], X[:, 1])
    # plt.show()


if __name__ == "__main__":
    print(linear_regression(array([[3, -5], [1, -7], [0, 1]]), array([0, 2, 5])))
    quit()

    svm_demo()
    # linear_regression_demo()
    quit()
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
