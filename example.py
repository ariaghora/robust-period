import numpy as np
import matplotlib.pyplot as plt
from robustperiod import robust_period, plot_robust_period
from robustperiod.utils import sinewave, triangle
from statsmodels.datasets.co2.data import load_pandas

if __name__ == '__main__':
    '''
    Dummy dataset
    '''
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

    '''
    CO_2 dataset
    '''
    co2 = load_pandas()

    # We only take 1000 samples due to numerical error by binom coef. computation
    # when N is large
    y_co2 = co2.data.fillna(method='ffill').values.squeeze()[:1000]
    plt.plot(y_co2)
    plt.title('$CO_2$ dataset')
    plt.show()

    lmb = 1e+6
    c = 2
    num_wavelets = 8
    zeta = 1.345
    periods, W, bivar, periodograms, p_vals, ACF = robust_period(
        y_co2, 'db10', num_wavelets, lmb, c, zeta)
    plot_robust_period(periods, W, bivar, periodograms, p_vals, ACF)
