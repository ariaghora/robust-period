import numpy as np
import matplotlib.pyplot as plt
from modwt import modwt
from statsmodels.tsa.filters.hp_filter import hpfilter

from utils import sinewave, triangle
from mperioreg import m_perio_reg


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
    - x: input signal with shape of (m, n), m is the number of observation and
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
    trend, y_hat = extract_trend(x, lmb)
    y_prime = residual_autocov(y_hat, c)

    plt.plot(x, label='Input')
    plt.plot(trend, label='Trend')
    plt.legend()
    plt.show()

    # 2) Decoupling multiple periodicities
    # ------------------------------------
    # Perform MODWT and ranking by robust wavelet variance

    W = modwt(y_prime, wavelet_method, level=num_wavelet)
    # TODO wavelet variance

    # 3) Robust single periodicity detection
    # --------------------------------------
    # Compute Huber periodogram and Huber ACF
    X = np.hstack([W, np.zeros_like(W)])

    # TODO implement concurrent periodogram extraction for several series
    periodograms = []
    for i, x in enumerate(X):
        print(f'Calculating periodogram for level {i+1}')
        periodograms.append(m_perio_reg(x))
    periodograms = np.array(periodograms)

    # plt.plot(periodogram[:1000])
    # plt.show()

    return (
        None,          # Periods
        W,             # Wavelets
        None,          # bivar
        periodograms,  # P
        None,          # pval
        None           # ACF
    )


def plot_robust_period(periods, W, bivar, periodograms, pval, ACF):
    nrows = W.shape[0]
    n_prime = periodograms.shape[1]
    fig, axs = plt.subplots(nrows, 3, sharex=True,
                            sharey=False, constrained_layout=True)

    for i in range(nrows):
        axs[i, 0].plot(W[i], color='green', linewidth=1)
        axs[i, 0].set(ylabel=f'Level {i+1}')
        axs[i, 0].set_title('Wavelet Coef: Var=NULL', fontsize=8)
        axs[i, 1].plot(periodograms[i][:n_prime//2], color='red', linewidth=1)
        axs[i, 1].set_title('Periodogram: p=NULL; per_T=NULL', fontsize=8)
        axs[i, 2].set_title(
            'ACF: acf_T=NULL; fin_T=NULL; Period=False', fontsize=8)
        for j in range(3):
            axs[i, j].tick_params(axis='both', which='major', labelsize=8)
            axs[i, j].tick_params(axis='both', which='minor', labelsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.show()


if __name__ == '__main__':
    m = 1000
    y1 = sinewave(m, 20, 1)
    y2 = sinewave(m, 50, 1)
    y3 = sinewave(m, 100, 1)
    tri = triangle(m, 10)
    noise = np.random.normal(0, 0.1, m)
    y = y1+y2+y3+tri+noise
    y[m//2] += 10  # sudden spike

    lmb = 1000000
    c = 2
    num_wavelets = 8

    res = robust_period(y, 'db4', num_wavelets, lmb, c)

    W = res[1]
    periodograms = res[3]

    plot_robust_period(None, W, None, periodograms, None, None)
