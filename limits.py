import matplotlib.pyplot as plt
import numpy as np


def plot_x_sin_1_over_x():
    y_values = []
    for x in reversed(np.arange(-1, 1, 0.005)):
        y = x * np.sin(1/x)
        y_values.append(y)
    plt.plot(y_values)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x * sin(1/x)")
    plt.show()


if __name__ == "__main__":
    plot_x_sin_1_over_x()
