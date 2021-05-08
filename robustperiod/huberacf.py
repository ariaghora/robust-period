import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def huber_acf(periodogram):
    N_prime = len(periodogram)
    N = N_prime // 2
    K = np.arange(N)

    cond_1 = periodogram[range(N)]
    cond_2 = (periodogram[2 * K] - periodogram[2 * K + 1]).sum() ** 2 / N_prime
    cond_3 = [periodogram[N_prime - k] for k in range(N + 1, N_prime)]

    P_bar = np.hstack([cond_1, cond_2, cond_3])
    P = np.real(np.fft.ifft(P_bar))

    denom = (N - np.arange(0, N)) * P[0]
    res = P[:N] / denom

    return res


def get_ACF_period(periodogram):
    N = len(periodogram)
    k = np.argmax(periodogram)
    res = huber_acf(periodogram)
    res_trim = res[:int(len(res) * 0.8)]  # The paper didn't use entire ACF
    res_scaled = 2 * ((res_trim - res_trim.min()) /
                      (res_trim.max() - res_trim.min())) - 1

    peaks, prop = find_peaks(res_scaled, height=0.5)
    distances = np.diff(peaks)

    acf_period = np.median(distances)
    acf_period = 0 if math.isnan(acf_period) else acf_period

    Rk = (0.5 * ((N/(k+1)) + (N/k)) - 1, 0.5 * ((N/k) + (N/(k-1))) + 1)
    final_period = acf_period if \
        (Rk[1] >= acf_period >= Rk[0]) else 0

    return acf_period, final_period, peaks


if __name__ == '__main__':
    periodograms = np.loadtxt('periodograms.csv', delimiter=',')

    per = periodograms[6]

    N = len(per)
    k = np.argmax(per)
    print(0.5 * ((N/(k+1)) + (N/k)) - 1, 0.5 * ((N/k) + (N/(k-1))) + 1)
    # print(N/k)

    res = huber_acf(per)

    _, _, peak_idx = get_ACF_period(per)

    plt.plot(res[:800])
    plt.scatter(peak_idx, res[:800][peak_idx], color='red')
    plt.show()
