#!/usr/bin/python

from numpy import sqrt
from termcolor import colored


def is_prime(n: int) -> bool:
    if n in [0, 1]:
        return False

    for i in range(2, int(sqrt(n)) + 1):
        if n % i == 0:
            return False

    return True


def number_six_pack_pattern():
    for i in range(1, 1000000):
        printable_number = colored(str(i), 'red') if is_prime(i) else colored(str(i), 'blue')
        print(printable_number, end=" ")
        if i % 6 == 0:
            print()


if __name__ == "__main__":
    number_six_pack_pattern()
