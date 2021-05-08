import numpy as np
import statsmodels.api as sm
from multiprocessing import Process, Queue, cpu_count


def get_fft_comp(series, j, t):
    n = len(series)
    w = 2. * np.pi * j / n
    idx_t = np.arange(0, n)
    MX = np.array([np.cos(w * idx_t), np.sin(w * idx_t)]).T
    if j != n/2:
        fitrob = sm.RLM(
            series,
            MX,
            M=sm.robust.norms.HuberT(t=t),
            deriv=0).fit()
        val = np.sqrt(n / (8 * np.pi)) * \
            np.complex(fitrob.params[0], -fitrob.params[1])
        return val
    else:
        fitrob = sm.RLM(
            series,
            MX,
            M=sm.robust.norms.HuberT(t=t),
            deriv=0).fit()
        val = np.sqrt(n / (2 * np.pi)) * \
            np.complex(fitrob.params[0], -0)
        return val


def process_chunk(pid, series, indices, t, out):
    fft = []
    for j in indices:
        fft.append(get_fft_comp(series, j, t))
    out.put({pid: fft})


def get_perio(series, t, n_process):
    n = len(series)
    g = n // 2
    idxs = np.array_split(np.arange(g), n_process)

    Q = Queue()
    procs = []
    for i in range(n_process):
        p = Process(target=process_chunk, args=[i, series, idxs[i], t, Q])
        procs.append(p)
        p.start()

    res_dict = {}
    for i in range(n_process):
        res_dict.update(Q.get())

    for p in procs:
        p.join()

    fft = []
    for k in range(n_process):
        fft += res_dict[k]

    perior = np.abs(np.ravel(fft)) ** 2
    return perior


def m_perio_reg(series, t=1.345, n_process=4):
    if n_process==-1:
        n_process = cpu_count() - 1
    perior = get_perio(series, t, n_process)
    return np.array(np.hstack([perior, np.flip(perior)]))
