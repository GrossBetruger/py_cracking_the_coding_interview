from collections import Counter, OrderedDict
from typing import Union, T

import matplotlib.pyplot as plt
import numpy as np
import termcolor
from numpy import ndarray
from numpy.random import binomial
from scipy.stats import binom


def simulate_game():
    """
    assuming a wager on coin tosses in which played 1 needs 2 more heads to win
    and player 2 needs 3 more tails to win (game will deterministically be settled in max of 4 turns
    """
    def normalize(num: str) -> str:
        if num.startswith("0b"):
            return normalize(num[2:])
        while len(num) < 4:
            return normalize("0" + num)

        return num.replace("0", "H").replace("1", "T")

    for i in range(0, 2**4):
        binary_representation = normalize(bin(i))

        color = "red" if len([c for c in binary_representation if c == "H"]) >= 2 else "blue"
        print(termcolor.cprint(binary_representation, color))


def count(random_var: Union[list, ndarray]) -> OrderedDict[T, float]:
    length = len(random_var)
    relative_frequencies = {k: v / length
                            for k, v in
                            Counter(random_var).items()}
    return OrderedDict(sorted(relative_frequencies.items()))


def binomial_distribution_simulation():
    binomial_sample = [binomial(10, 0.3)
                       for _ in range(10000)]
    observed_frequencies = count(binomial_sample)
    binomial_distribution = binom(10, 0.3)
    for num_of_successes in observed_frequencies:
        allowed_error = 0.1
        phi = binomial_distribution.pmf(num_of_successes)
        within_error_range = np.allclose(observed_frequencies[num_of_successes], phi,
                                         rtol=allowed_error)
        print(num_of_successes, observed_frequencies[num_of_successes], phi, within_error_range)

    plt.plot(binomial_distribution.pmf(range(0, 10)), 'o', color='red')
    plt.bar(list(observed_frequencies), list(observed_frequencies.values()))
    plt.show()


def standard_normal_distribution(x: float) -> float:
    """
    :param x: value x
    :return: phi(x) in the standard normal distribution
    """
    stigler_formula = (np.e ** -x ** 2) / np.sqrt(np.pi)
    regular_formula = (np.e**((-x**2)/2)) / np.sqrt(2 * np.pi)
    return regular_formula


if __name__ == "__main__":
    # Creation standard normal distribution (mu=0, variance=1)
    standard_range = np.arange(-3, 3, 0.00001)
    sample = []
    for n in standard_range:
        if standard_normal_distribution(n) > np.random.uniform(low=0, high=1):
            sample.append(n)
    plt.hist(sample, bins=30)
    plt.show()

    # Simulate binomial experiment and compare with PMF values
    binomial_distribution_simulation()

    # simulate_game()
