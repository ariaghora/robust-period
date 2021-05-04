import numpy as np


def sinewave(N, period, amplitude):
    x1 = np.arange(0, N, 1)
    frequency = 1/period
    theta = 0
    y = amplitude * np.sin(2 * np.pi * frequency * x1 + theta)
    return y


def triangle(length, amplitude):
    mid = length // 2
    return np.hstack([np.linspace(0, amplitude, mid),
                      np.linspace(amplitude, 0, length-mid)])
