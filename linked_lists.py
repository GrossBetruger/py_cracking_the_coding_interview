from collections import deque
from collections import Counter
from typing import List, TypeVar

T = TypeVar('T')


# Chapter 2 Linked Lists
# question 2.1

def remove_dups(linked_list: deque) -> deque:
    def dedup(ll: deque):
        counter = Counter()
        for item in ll:
            counter[item] += 1

        result = deque()
        ll.reverse()
        for item in ll:
            if counter[item] == 1:
                result.append(item)
            else:
                counter[item] -= 1
        result.reverse()
        return result

    return dedup(linked_list)


# question 2.2
def remove_dedup_no_extra_buffer(linked_list: list):
    to_remove = list()
    for i, item in enumerate(linked_list):
        for j, next_item in enumerate(linked_list[i + 1:]):
            if next_item == item:
                to_remove.append(i + j + 1)
                break

    offset = 0
    for i in to_remove:
        del linked_list[i + offset]

        offset -= 1

    return linked_list


# question 2.3
def kth_to_last(linked_list: List[T], k: int) -> T:
    """
    :param linked_list: list from which returned element is extracted:
    :param k: offset (from last) of returned element:
    :return: element of generic type 'T'

    big(O) complexity: 2N (N)
    """
    length = 0
    for i, _ in enumerate(linked_list):
        length = i + 1

    for i, item in enumerate(linked_list):
        if length -1 - k == i:
            return item


# question 2.4
def partition(linked_list: List, partition_element: int):
    result = list()
    for elem in linked_list:
        if elem < partition_element:
            result.append(elem)

    for elem in linked_list:
        if elem >= partition_element:
            result.append(elem)

    return result


# question 2.5
def sum_lists(lst1: List, lst2: List) -> List:
    """
    :param lst1 first number 1's digit in head:
    :param lst2: second number 1's digit in head
    :return sum of numbers as list of digits (1's digit in head):
    """
    result = list()
    carry = 0

    def pad_zeros(small_lst, big_lst):
        if len(small_lst) > len(big_lst):
            small_lst, big_lst = big_lst, small_lst
        for _ in range(len(big_lst) - len(small_lst)):
            small_lst.append(0)

    pad_zeros(lst1, lst2)

    for i, j in zip(lst1, lst2):
        digit = (i + j + carry) % 10
        carry = 1 if i + j + carry >= 10 else 0
        result.append(digit)

    if carry == 1:
        result.append(carry)

    return result


# question 2.6
def palindrome(l: list) -> bool:
    """
    :param l list to check if palindrome:
    :return True if palindrome else False:
    """

    def method1():
        length = len(l)
        # check til midway (no need to check middle if exists e.g length 5, check 0, 1 == -1, -2
        for i in range(length // 2):
            a, z = l[i], l[-i-1]
            if a != z:
                return False
        return True

    def method2():
        return l == list(reversed(l))

    def method3():
        stack = list()
        for link in l:
            stack.append(link)
        for link in l:
            if link != stack.pop():
                return False
        return True

    predicates = [method1(), method2(), method3()]
    assert set(predicates) == {False} or set(predicates) == {True}  # all False or all True
    return all(predicates)


if __name__ == "__main__":
    print(remove_dups(deque([1, 3, 2, 1, 2, 2])))
    print(remove_dedup_no_extra_buffer([1, 3, 2, 1, 2, 2]))