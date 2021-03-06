import unittest
from random import randint

import numpy as np

from collections import deque

import scipy
from numpy import array

from error_correction import HammingMessage
from linear_algebra import gaussian_elimination, plu_decomposition, rotate_right, rotate_left, \
    two_d_vector_from_magnitude_and_angle, Orientation, multiply_matrices, LeastSquares, take_derivative, \
    linear_regression
from linked_lists import remove_dups, remove_dedup_no_extra_buffer, kth_to_last, partition, sum_lists, palindrome
from stats_and_probability import standard_normal_distribution


class UnitTests(unittest.TestCase):
    def test_linked_list_dedup_with_extra_buffer(self):
        self.assertEqual(deque([1, 3, 2]), remove_dups(deque([1, 3, 2, 1, 2, 2])))
        self.assertEqual(deque([7, 77]), remove_dups(deque([7, 77, 7, 77, 7, 77])))
        self.assertEqual(deque([77, 7]), remove_dups(deque([77, 7, 7, 77, 7, 77])))

    def test_linked_list_dedup_no_extra_buffer(self):
        self.assertEqual([1, 3, 2], remove_dedup_no_extra_buffer([1, 3, 2, 1, 2, 2]))
        self.assertEqual([7, 77], remove_dedup_no_extra_buffer([7, 77, 7, 77, 7, 77]))
        self.assertEqual([77, 7], remove_dedup_no_extra_buffer([77, 7, 7, 77, 7, 77]))

    def test_linked_list_kth_to_last_element(self):
        self.assertEqual(2, kth_to_last(linked_list=[1, 2, 3], k=1))
        self.assertEqual(1, kth_to_last(linked_list=[1, 2, 3], k=2))
        self.assertEqual(3, kth_to_last(linked_list=[1, 2, 3], k=0))
        self.assertEqual(0, kth_to_last(linked_list=[1, 1, 1, 0, 1, 1], k=2))

    def test_linked_list_partition(self):
        self.assertEqual([3, 2, 1, 5, 8, 5, 10], partition([3, 5, 8, 5, 10, 2, 1], partition_element=5))
        self.assertEqual([1, 2, 2, 4], partition([1, 2, 4, 2], partition_element=3))

    def test_linked_list_sum_lists(self):
        # 617 + 295 = 912
        self.assertEqual([2, 1, 9], sum_lists([7, 1, 6], [5, 9, 2]))

        # 999 + 3 = 1002
        self.assertEqual([2, 0, 0, 1], sum_lists([9, 9, 9], [3]))

        # 504 + 2001 = 2505
        self.assertEqual([5, 0, 5, 2], sum_lists([4, 0, 5], [1, 0, 0, 2]))

        def normalize(n: int) -> list:
            return list(reversed([int(x) for x in list(str(n))]))

        def reverse_normalize(l: list) -> int:
            return int("".join([str(x) for x in reversed(l)]))

        for a, b, s in [
            (537, 579, 1116),
            (283, 795, 1078),
            (1974, 3050, 5024),
            (4384, 1345, 5729)
        ]:
            self.assertEqual(s, a + b)
            a, b = normalize(a), normalize(b)
            normalized_sum = sum_lists(a, b)
            self.assertEqual(s, reverse_normalize(normalized_sum))

    def test_palindrome(self):
        palindrome_lst = [3, 4, 5, 4, 3]
        self.assertTrue(palindrome(palindrome_lst))

        not_palindrome_lst = [3, 4, 5, 4, 2]
        self.assertFalse(palindrome(not_palindrome_lst))

        palindrome_lst = [2, 1, 1, 2]
        self.assertTrue(palindrome(palindrome_lst))

        not_palindrome_lst = [1, 1, 1, 0]
        self.assertFalse(palindrome(not_palindrome_lst))

        not_palindrome_lst = [2, 1, 0, 3, 4, 0, 1, 2]
        self.assertFalse(palindrome(not_palindrome_lst))

    def test_gaussian_elimination(self):
        coefficients_matrix = [[-1, 3], [3, 5]]
        b_vector = array([3, 7])
        solutions = gaussian_elimination(coefficients_matrix, b_vector)
        expected_solutions = array([3 / 7, 8 / 7])
        self.assertTrue(np.allclose(expected_solutions, solutions))

    def test_plu_decomposition(self):
        matrix = array([[1, 2], [3, 4]])
        p, l, u = plu_decomposition(matrix)
        self.assertTrue(np.allclose(np.dot(np.dot(p, l), u), matrix))

    def test_rotate_right(self):
        i_basis_vector = (1, 0)
        j_basis_vector = (0, 1)
        i_prime_vector = (0, -1)
        j_prime_vector = (1, 0)
        self.assertTrue(np.allclose(i_prime_vector, rotate_right(i_basis_vector)))
        self.assertTrue(np.allclose(j_prime_vector, rotate_right(j_basis_vector)))

    def test_rotate_left(self):
        i_basis_vector = (1, 0)
        j_basis_vector = (0, 1)
        i_prime_vector = (0, 1)
        j_prime_vector = (-1, 0)
        self.assertTrue(np.allclose(i_prime_vector, rotate_left(i_basis_vector)))
        self.assertTrue(np.allclose(j_prime_vector, rotate_left(j_basis_vector)))

        # 4 rotation are equal to no rotation
        self.assertTrue(np.allclose(i_basis_vector, rotate_left(rotate_left(rotate_left(rotate_left(i_basis_vector))))))

    def test_2d_vec_trigonometric_creation(self):
        v = two_d_vector_from_magnitude_and_angle(magnitude=7, angle=35, orientation=Orientation.XPositiveYNegative)
        x, y = v
        self.assertAlmostEqual(5.73406431, x)
        self.assertAlmostEqual(-4.01503505, y)

        v = two_d_vector_from_magnitude_and_angle(magnitude=8, angle=10, orientation=Orientation.XPositiveYPositive)
        x, y = v
        self.assertAlmostEqual(1.38918542, x)
        self.assertAlmostEqual(7.87846202, y)

        v = two_d_vector_from_magnitude_and_angle(magnitude=4, angle=310, orientation=Orientation.Absolute)
        w = two_d_vector_from_magnitude_and_angle(magnitude=7, angle=50, orientation=Orientation.Absolute)
        x, y = v
        self.assertAlmostEqual(2.57115044, x)
        self.assertAlmostEqual(-3.06417777, y)
        x, y = w
        self.assertAlmostEqual(4.49951327, x)
        self.assertAlmostEqual(5.3623111, y)

    def test_matrix_multiplication(self):
        a = array([[5, -2], [-1, 5]])
        b = array([[2, 0], [0, 2]])
        expected = np.array([[10., -4.],
                             [-2., 10.]])
        self.assertTrue(np.allclose(expected, multiply_matrices(a, b)))

        a = array([[1, -1, 5], [5, 5, 0]])
        b = array([[-2, 1], [5, 2], [5, -2]])
        expected = np.array([[18., -11.],
                             [15., 15.]])
        self.assertTrue(np.allclose(expected, multiply_matrices(a, b)))

        a = array([[2, 8, 3], [5, 4, 1]])
        b = array([[4, 1], [6, 3], [2, 4]])

        expected = np.array([[62., 38.],
                             [46., 21.]])
        self.assertTrue(np.allclose(expected, multiply_matrices(a, b)))

        a = array([[1, 2], [-2, 3]])
        b = array([[0, -1, 5], [3, 2, 1]])
        expected = np.array([[6., 3., 7.],
                             [9., 8., -7.]])
        self.assertTrue(np.allclose(expected, multiply_matrices(a, b)))

    def test_least_squares(self):
        x = array([1, 2, 3, 4, 5, 6, 7])
        y = array([1.5, 3.8, 6.7, 9.0, 11.2, 13.6, 16])
        least_squares = LeastSquares()
        least_squares.train(x, y)

        self.assertLessEqual(4.0000000000000036, least_squares.predict(2))
        self.assertLessEqual(11.242857142857142, least_squares.predict(5))
        self.assertLessEqual(16.07142857142857, least_squares.predict(7))

        # compare to matrix method
        coefficients = array([x, np.ones(x.size)]).transpose()
        m, b = linear_regression(coefficients, y)
        predict = lambda _x: m*_x + b
        self.assertLessEqual(4.0000000000000036, predict(2))
        self.assertLessEqual(11.242857142857142, predict(5))
        self.assertLessEqual(16.07142857142857, predict(7))

    def test_derivative_matrix(self):
        polynomial = array([5, 4, 5, 1])  # representing (1x^3 + 5x^2 + 4x + 5)
        expected_derivative = [4., 10., 3., 0.]  # representing d/dx(1x^3 + 5x^2 + 4x + 5) = 3x^2 + 10x +4
        self.assertTrue(np.allclose(expected_derivative, take_derivative(polynomial)))

    def test_linear_regression_matrix_method(self):
        observed_x_values = array([[1, 1],
                                   [2, 1],
                                   [3, 1]])
        observed_y_values = array([2, 1, 3])

        expected_solution_vector = array([0.5, 1. ])

        self.assertTrue(
            np.allclose(expected_solution_vector,
                        linear_regression(observed_x_values, observed_y_values)
                        )
        )

    def test_hamming_error_correction(self):
        message_size = 16
        assert np.math.log(message_size, 2) - int(np.math.log(message_size, 2)) == 0
        random_message = np.random.randint(0, 2, message_size)
        hamming = HammingMessage(random_message)
        hamming.prepare()
        for corrupted_bit_offset in [1, 4, 5, 7]:
            hamming.message[corrupted_bit_offset] = not hamming.message[corrupted_bit_offset]
            self.assertEqual(corrupted_bit_offset, hamming.check())
            hamming.prepare()  # fix corrupted parity bit

    def test_standard_normal_distribution(self):
        for x in np.arange(-3, 3, 0.1):
            self.assertAlmostEqual(standard_normal_distribution(x), scipy.stats.norm.pdf(x))

    def test_normal_distribution_pdf(self):
        integral = float()
        # use standard normal dist function as pdf for normal dist (they are equivalent)
        pdf = standard_normal_distribution

        prev_x = 0.
        for x in np.arange(-100, 100, 0.01):
            dx = x - prev_x
            integral += pdf(x) * dx
            prev_x = x
        epsilon = 0.000000000000001
        self.assertAlmostEqual(1., integral, delta=epsilon,
                               msg="the integral of pdf(x)dx over R should be 1")


if __name__ == '__main__':
    unittest.main()
