import numpy as np
import statsmodels.api as sm


def m_perio_reg(series, t=1.345):
    n = len(series)
    g = n // 2
    perior = []
    fft = []

    for j in range(g):
        w = 2. * np.pi * j / n
        X1 = []
        X2 = []
        for i in range(n):
            X1.append(np.cos(w * i))
            X2.append(np.sin(w * i))

        if j != n/2:
            MX = np.array([X1, X2]).T
            fitrob = sm.RLM(
                series,
                MX - 1,
                M=sm.robust.norms.HuberT(t=t),
                deriv=0).fit()
            val = np.sqrt(n / (8*np.pi)) * \
                np.complex(fitrob.params[0], -fitrob.params[1])
            fft.append(val)
        else:
            MX = np.array([X1, X2]).T
            fitrob = sm.RLM(
                series,
                MX - 1,
                M=sm.robust.norms.HuberT(t=t)).fit()
            val = np.sqrt(n / (2*np.pi)) * \
                np.complex(fitrob.params[0], -0)
            fft.append(val)

        perior.append(np.abs(fft[j]) ** 2)

    if n % 2 != 0:
        return np.array(perior + perior[::-1])
    else:
        return np.delete(perior + perior[::-1], g)
