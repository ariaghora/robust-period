import numpy as np
import scipy


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def p_val_g_stat(g0, N, method='author'):
    if g0 == 0:
        g0 = 1e-8

    k1 = int(np.floor(1/g0))
    terms = np.arange(1, k1+1, dtype='int32')
    # Robust Period Equation

    fac = np.math.factorial
    binom = scipy.special.binom

    def event_term(k, N=N, g0=g0):
        return (-1)**(k-1) * binom(N, k) * (1-k*g0)**(N-1)
    # R fisher test equation

    def r_event_term(k, N, g0):
        temp_x = float(choose(N, k))
        temp_y = 1-k*g0
        if temp_y == 0:
            temp_y += 1e-8
        temp_z = np.log(temp_x) + (N-1) * np.log(temp_y)
        return (-1)**(k-1) * np.exp(temp_z)

    if method == 'author':
        vect_event_term = np.vectorize(event_term)
    else:
        vect_event_term = np.vectorize(r_event_term)

    pval = sum(vect_event_term(terms, N, g0))
    if pval > 1:
        pval = 1
    return pval


def fisher_g_test(per, method='author'):
    ''' per: periodogram'''
    g = max(per) / np.sum(per)
    pval = p_val_g_stat(g, len(per), method=method)
    return pval, g


if __name__ == '__main__':
    periodograms = np.loadtxt('periodograms.csv', delimiter=',')

    p, g = fisher_g_test(periodograms[1])
    print(p)
