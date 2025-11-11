import numpy as np
from ..dsp.utils import shift_signal

def auto_align_lag(x, y, maxlag_samples):
    m = int(maxlag_samples)
    if m <= 0: return 0
    N = min(len(x), 10000)
    xa = x[:N] - np.mean(x[:N]); ya = y[:N] - np.mean(y[:N])
    lags = np.arange(-m, m+1)
    corr = [np.dot(xa[:N-abs(L)], ya[abs(L):N]) if L>=0
            else np.dot(xa[abs(L):N], ya[:N-abs(L)]) for L in lags]
    return int(lags[int(np.argmax(corr))])

def solver_scalar(x, ambient, fs, calib_sec=2.0, auto_align=False, maxlag_ms=10.0):
    anti = -ambient.astype(np.float32)
    lag = 0
    if auto_align:
        lag = auto_align_lag(x, anti, int(round(maxlag_ms*1e-3*fs)))
        anti = shift_signal(anti, lag)
    N = min(len(x), int(calib_sec*fs))
    if N > 100:
        g = float(-np.dot(x[:N], ambient[:N]) /
                  (np.dot(ambient[:N], ambient[:N]) + 1e-12))
    else:
        g = 1.0
    anti *= g
    return anti.astype(np.float32), {'gain': g, 'lag': lag}
