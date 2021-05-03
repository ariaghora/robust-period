import numpy as np
import matplotlib.pyplot as plt
from modwt import modwt
from statsmodels.tsa.filters.hp_filter import hpfilter
from astropy.stats import biweight_midvariance

from utils import sinewave, triangle
from mperioreg import m_perio_reg
from huberacf import huber_acf, huber_acf_2


def extract_trend(y, reg):
    _, trend = hpfilter(y, reg)
    y_hat = y - trend
    return trend, y_hat


def huber_func(x, c):
    return np.sign(x) * np.minimum(np.abs(x), c)


def MAD(x):
    return np.mean(np.abs(x - np.mean(x)))


def residual_autocov(x, c):
    '''
    The \Psi transformation function
    '''
    mu = np.median(x)
    s = MAD(x)
    return huber_func((x - mu)/s, c)


def robust_period(x, wavelet_method, num_wavelet, lmb, c, zeta=1.345):
    '''
    Params:
    - x: input signal with shape of (m, n), m is the number of observation and
         n is the number of series
    - wavelet_method: 
    - num_wavelet:
    - lmb: Lambda (regularization param) in Hodrick–Prescott (HP) filter
    - c: Huber function hyperparameter
    - zeta: M-Periodogram hyperparameter

    Returns:
    Array of periods
    '''

    assert wavelet_method.startswith('db'), \
        'wavelet method must be Daubechies family, e.g., db1, ..., db34'

    # 1) Preprocessing
    # ----------------
    # Extract trend and then deterend input series. Then perform residual
    # autocorrelation to remove extreme outliers.
    trend, y_hat = extract_trend(x, lmb)
    y_prime = residual_autocov(y_hat, c)

    # 2) Decoupling multiple periodicities
    # ------------------------------------
    # Perform MODWT and ranking by robust wavelet variance
    W = modwt(y_prime, wavelet_method, level=num_wavelet)

    # compute wavelet variance for all levels
    bivar = np.array([biweight_midvariance(w) for w in W]).reshape(-1, 1)

    # 3) Robust single periodicity detection
    # --------------------------------------
    # Compute Huber periodogram
    X = np.hstack([W, np.zeros_like(W)])

    # TODO implement concurrent periodogram extraction for several series
    periodograms = []
    for i, x in enumerate(X):
        print(f'Calculating periodogram for level {i+1}')
        periodograms.append(m_perio_reg(x))
    periodograms = np.array(periodograms)

    # TODO Compute p-value

    # TODO remove this line at the end
    np.savetxt('periodograms.csv', periodograms, delimiter=',')

    # Compute Huber ACF
    ACF = []
    for periodogram in periodograms:
        ACF.append(huber_acf(periodogram))
    ACF = np.array(ACF)

    return (
        None,          # Periods
        W,             # Wavelets
        bivar,         # bivar
        periodograms,  # P
        None,          # pval
        ACF            # ACF
    )


def plot_robust_period(periods, W, bivar, periodograms, pval, ACF):
    nrows = W.shape[0]
    n_prime = periodograms.shape[1]
    fig, axs = plt.subplots(nrows, 3, sharex=False,
                            sharey=False, constrained_layout=True)

    per_Ts = (n_prime / periodograms.argmax(1)).astype(int)

    ACF = ACF[:, :int(0.8 * (n_prime//2))]
    ACF = 2 * ((ACF - ACF.min(1, keepdims=True)) /
               (ACF.max(1, keepdims=True) - ACF.min(1, keepdims=True))) - 1

    for i in range(nrows):
        axs[i, 0].plot(W[i], color='green', linewidth=1)
        axs[i, 0].set(ylabel=f'Level {i+1}')
        axs[i, 0].set_title('Wavelet Coef: Var=0', fontsize=8)
        axs[i, 1].plot(periodograms[i][:n_prime//2], color='red', linewidth=1)
        axs[i, 1].set_title(f'Periodogram: p=0; per_T={per_Ts[i]}', fontsize=8)
        axs[i, 2].plot(ACF[i], color='blue', linewidth=1)
        axs[i, 2].set_title(
            'ACF: acf_T=0; fin_T=0; Period=False', fontsize=8)
        # axs[i, 2].set_ylim((-1, 1))
        for j in range(3):
            axs[i, j].tick_params(axis='both', which='major', labelsize=8)
            axs[i, j].tick_params(axis='both', which='minor', labelsize=8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.show()

    plt.plot(bivar, linestyle='dashed', marker='s',
             label='robust wavelet variance')
    plt.xlabel('Wavelet level')
    plt.ylabel('Wavelet variance')
    plt.legend()
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

    lmb = 10000000
    c = 2
    num_wavelets = 8
    zeta = 1.345

    res = robust_period(y, 'db10', num_wavelets, lmb, c, zeta)

    _, W, bivar, periodograms, _, ACF = res

    plot_robust_period(None, W, bivar, periodograms, None, ACF)
