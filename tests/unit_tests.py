import unittest
import numpy as np


from collections import deque
from numpy import array
from linear_algebra import gaussian_elimination, plu_decomposition, rotate_right, rotate_left
from linked_lists import remove_dups, remove_dedup_no_extra_buffer, kth_to_last, partition


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

    def test_gaussian_elimination(self):
        coefficients_matrix = [[-1, 3], [3, 5]]
        b_vector = array([3, 7])
        solutions = gaussian_elimination(coefficients_matrix, b_vector)
        expected_solutions = array([3/7, 8/7])
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


if __name__ == '__main__':
    unittest.main()
