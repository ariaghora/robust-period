import numpy as np
import matplotlib.pyplot as plt
from modwt import modwt
from statsmodels.tsa.filters.hp_filter import hpfilter


from utils import sinewave, triangle


def extract_trend(y, reg):
    _, trend = hpfilter(y, reg)
    y_hat = y - trend
    return trend, y_hat


def huber_func(x, c):
    return np.sign(x) * np.minimum(np.abs(x), c)


def residual_autocov(x, c):
    '''
    The \Psi transformation function
    '''
    mu = np.median(x)
    s = np.mean(np.abs(x - np.mean(x)))
    return huber_func((x - mu)/s, c)


def robust_period(x, wavelet_method, num_wavelet, lmb, c):
    '''
    Params:
    - y: input signal with shape of (m, n), m is the number of observation and
         n is the number of series
    - wavelet_method: 
    - num_wavelet:
    - lmb: Lambda (regularization param) in Hodrick–Prescott (HP) filter
    - c: Huber function hyperparameter

    Returns:
    Array of periods
    '''

    # 1) Preprocessing
    # ----------------
    # Extract trend and then deterend input series. Then perform residual
    # autocorrelation to remove extreme outliers.
    trend, y_hat = extract_trend(y, lmb)
    y_prime = residual_autocov(y_hat, c)


    # 2) Decoupling multiple periodicities
    # ------------------------------------
    # Perform MODWT and ranking by robust wavelet variance

    W = modwt(y_prime, wavelet_method, num_wavelet)
    # TODO wavelet variance


    # 3) Robust single periodicity detection
    # --------------------------------------
    # Compute Huber periodogram and Huber ACF


    

    plt.plot(W.T[:,-1], label='y_hat')
    # plt.plot(y_prime, label='y_prime')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    y1 = sinewave(1000, 20, 1)
    y2 = sinewave(1000, 50, 1)
    y3 = sinewave(1000, 100, 1)
    tri = triangle(1000, 10)
    noise = np.random.normal(0, 0.1, 1000)
    y = y1+y2+y3+tri+noise
    y[500] += 10  # sudden spike

    lmb = 5000
    c = 2

    robust_period(y, 'db1', 7, lmb, c)
