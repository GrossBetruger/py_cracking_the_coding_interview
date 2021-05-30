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


if __name__ == "__main__":
    print(remove_dups(deque([1, 3, 2, 1, 2, 2])))
    print(remove_dedup_no_extra_buffer([1, 3, 2, 1, 2, 2]))