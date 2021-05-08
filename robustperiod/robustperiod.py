import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import biweight_midvariance
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy.signal import find_peaks

from .modwt import modwt
from .utils import sinewave, triangle
from .mperioreg import m_perio_reg
from .huberacf import huber_acf, get_ACF_period
from .fisher import fisher_g_test


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


def robust_period_full(x, wavelet_method, num_wavelet, lmb, c, zeta=1.345):
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
    - Array of periods
    - Wavelets
    - bivar
    - Periodograms
    - pval
    - ACF
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
    # TODO Clarifying Lj, so we can omit first Lj from wj
    bivar = np.array([biweight_midvariance(w) for w in W])

    # 3) Robust single periodicity detection
    # --------------------------------------
    # Compute Huber periodogram
    X = np.hstack([W, np.zeros_like(W)])

    periodograms = []
    p_vals = []
    for i, x in enumerate(X):
        print(f'Calculating periodogram for level {i+1}')
        perio = m_perio_reg(x, n_process=-1)
        p_val, _ = fisher_g_test(perio)
        periodograms.append(perio)
        p_vals.append(p_val)
    periodograms = np.array(periodograms)
    # np.savetxt('periodograms.csv', periodograms, delimiter=',')

    # Compute Huber ACF
    ACF = np.array([huber_acf(p) for p in periodograms])

    periods = []
    for acf in ACF:
        peaks, _ = find_peaks(acf)
        distances = np.diff(peaks)
        final_period = np.median(distances)
        periods.append(final_period)
    periods = np.array(periods)

    periods = []
    for p in periodograms:
        _, final_period, _ = get_ACF_period(p)
        periods.append(final_period)
    periods = np.array(periods)
    final_periods = np.unique(periods[periods > 0])

    return (
        final_periods,  # Periods
        W,              # Wavelets
        bivar,          # bivar
        periodograms,   # periodograms
        p_vals,         # pval
        ACF             # ACF
    )


def robust_period(x, wavelet_method, num_wavelet, lmb, c, zeta=1.345):
    assert x.ndim == 2, 'Input must be a 2D array'

    periods_list = []
    for i in range(x.shape[1]):
        res = robust_period_full(x[:, i], wavelet_method, num_wavelet, lmb, c, zeta)
        periods_list.append(res[0])
    return np.array(periods_list)


def plot_robust_period(periods, W, bivar, periodograms, p_vals, ACF):
    nrows = W.shape[0]
    n_prime = periodograms.shape[1]

    per_Ts = (n_prime / periodograms.argmax(1)).astype(int)

    ACF = ACF[:, :int(0.8 * (n_prime//2))]
    ACF = 2 * ((ACF - ACF.min(1, keepdims=True)) /
               (ACF.max(1, keepdims=True) - ACF.min(1, keepdims=True))) - 1

    acf_periods = []
    final_periods = []
    peaks_arr = []
    for p in periodograms:
        acf_period, final_period, peaks = get_ACF_period(p)
        acf_periods.append(acf_period)
        final_periods.append(final_period)
        peaks_arr.append(peaks)

    fig, axs = plt.subplots(nrows, 3, sharex=False, constrained_layout=True)
    for i in range(nrows):
        axs[i, 0].plot(W[i], color='green', linewidth=1)
        axs[i, 0].set(ylabel=f'Level {i+1}')
        axs[i, 0].set_title(f'Wavelet Coef: Var={bivar[i]:.3f}', fontsize=8)
        axs[i, 1].plot(periodograms[i][:n_prime//2], color='red', linewidth=1)
        axs[i, 1].set_title(
            f'Periodogram: p={p_vals[i]:.2e}; per_T={per_Ts[i]}', fontsize=8)
        axs[i, 2].plot(ACF[i], color='blue', linewidth=1)
        axs[i, 2].axhline(0.5, color='green')

        has_period = final_periods[i] > 0
        if has_period:
            axs[i, 2].scatter(peaks_arr[i], ACF[i]
                              [peaks_arr[i]], marker='*', color='red')

        axs[i, 2].set_title(
            f'ACF: acf_T={acf_periods[i]}; fin_T={final_periods[i]}; Period={has_period}', fontsize=8)

        for j in range(3):
            axs[i, j].tick_params(axis='both', which='major', labelsize=8)
            axs[i, j].tick_params(axis='both', which='minor', labelsize=8)
            axs[i, j].yaxis.get_label().set_fontsize(8)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    plt.show()

    plt.plot(np.arange(nrows)+1, bivar, linestyle='dashed', marker='s',
             label='Wavelet variance')
    plt.xlabel('Wavelet level')
    plt.ylabel('Wavelet variance')
    plt.legend()
    plt.show()
