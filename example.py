import numpy as np
from robustperiod import robust_period, plot_robust_period
from robustperiod.utils import sinewave, triangle

if __name__ == '__main__':
    m = 1000
    y1 = sinewave(m, 20, 1)
    y2 = sinewave(m, 50, 1)
    y3 = sinewave(m, 100, 1)
    tri = triangle(m, 10)
    noise = np.random.normal(0, 0.1, m) 
    y = y1+y2+y3+tri+noise
    y[m//2] += 10  # sudden spike

    lmb = 1e+6
    c = 2
    num_wavelets = 8
    zeta = 1.345

    periods, W, bivar, periodograms, p_vals, ACF = robust_period(
        y, 'db10', num_wavelets, lmb, c, zeta)

    plot_robust_period(periods, W, bivar, periodograms, p_vals, ACF)