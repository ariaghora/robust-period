import numpy as np
import statsmodels.api as sm


def m_perio_reg(series, t=1.345):
    n = len(series)
    g = n // 2
    fft = []

    for j in range(g):
        w = 2. * np.pi * j / n
        idx = np.arange(0, n).reshape(-1, 1)
        MX = np.hstack([np.cos(w * idx), np.sin(w * idx)])
        if j != n/2:
            fitrob = sm.RLM(
                series,
                MX,
                M=sm.robust.norms.HuberT(t=t),
                deriv=0).fit()
            val = np.sqrt(n / (8 * np.pi)) * \
                np.complex(fitrob.params[0], -fitrob.params[1])
            fft.append(val)
        else:
            fitrob = sm.RLM(
                series,
                MX,
                M=sm.robust.norms.HuberT(t=t),
                deriv=0).fit()
            val = np.sqrt(n / (2 * np.pi)) * \
                np.complex(fitrob.params[0], -0)
            fft.append(val)

    perior = np.abs(fft) ** 2

    return np.array(np.hstack([perior, np.flip(perior)]))
