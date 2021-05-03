import numpy as np
import matplotlib.pyplot as plt


def huber_acf(periodogram):
    N_prime = len(periodogram)
    N = N_prime // 2

    cond_1 = periodogram[range(N)]

    tot = 0
    for k in range(N):
        tot += (periodogram[2 * k] - periodogram[2 * k + 1])
    cond_2 = [(tot ** 2) / N_prime]

    cond_3 = [periodogram[N_prime - k] for k in range(N + 1, N_prime)]

    P_bar = np.hstack([cond_1, cond_2, cond_3])
    P = np.fft.ifft(P_bar)

    denom = (N - np.arange(0, N)) * P[0]
    res = P[:N] / denom

    # return 2 * ((res - res.min()) / (res.max() - res.min())) - 1
    return res


def gmat(n):
    def Gj(j, n):
        icomp = complex(0, 1)
        w = icomp * 2 * np.pi * (np.arange(n) * j) / n
        return w
    mat_aux = np.array([np.exp(Gj(i, n)) for i in range(n)])

    return (1/np.sqrt(n)) * mat_aux


def huber_acf_2(periodogram):
    # periodogram = periodogram[:len(periodogram)//2]
    sample_t = len(periodogram)
    dmatrix = np.diag(periodogram)
    gmatrix = gmat(sample_t)
    covmat = 2 * np.pi * np.real(np.conj(gmatrix.T) @ dmatrix @ gmatrix)

    return covmat[0, :sample_t]


if __name__ == '__main__':
    periodograms = np.loadtxt('periodograms.csv', delimiter=',')

    res = huber_acf(periodograms[3][:])

    plt.plot(res[:800])
    plt.show()
