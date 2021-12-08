import numpy as np

from math import pi
from termcolor import colored
from os import cpu_count
from multiprocessing.pool import Pool
from typing import Callable, Optional


def rand_neg(num: float):
    if np.random.choice([True, False]):
        num = -num
    return num


def multi_proc_mc_simulator(n: int, f: Callable, args: Optional[tuple] = None):
    task_size = n // cpu_count()
    num_tasks = n // task_size
    pool = Pool(cpu_count())
    task = (task_size,) if args is None else (task_size,) + args
    results = pool.map(f, [task] * num_tasks)
    return np.mean(results)


def circle_in_square_dart_throw(args: tuple):
    n, = args
    dart_xs_vec = np.random.rand(n) * np.random.choice([-1, 1], n)
    dart_ys_vec = np.random.rand(n) * np.random.choice([-1, 1], n)
    points = dart_xs_vec**2 + dart_ys_vec**2
    hits = np.count_nonzero(points <= 1)
    return (hits / n) * 4


def two_strikes_and_you_are_out(p: float):
    strikes = int()
    num_games = int()
    while True:
        if strikes == 2:
            return num_games
        if np.random.rand() < p:
            strikes = 0
        else:
            strikes += 1
        num_games += 1


def mc_simulate_two_strikes(args: tuple) -> float:
    n, p = args
    results = [two_strikes_and_you_are_out(p) for _ in range(n)]
    return float(np.mean(results))


if __name__ == '__main__':
    print("simulating two strikes and you're out game...")
    n = 10_000_000
    # result should be 6 as the analytical expectation of this game is (2-p)/(1-p)^2
    print('expectation of game:', multi_proc_mc_simulator(n, mc_simulate_two_strikes, args=(0.5,)))
    print()
    print('simulating dart throwing...')
    pi_approx = multi_proc_mc_simulator(300_000_000, circle_in_square_dart_throw)

    # color pi digits we got correct
    pi_approx_highlighted = str()
    for c1, c2 in zip(str(pi_approx), str(pi)):
        c = colored(c1, 'green') if c1 == c2 else colored(c1, 'red')
        pi_approx_highlighted += c

    print(pi_approx_highlighted)
