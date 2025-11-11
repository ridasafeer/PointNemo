import numpy as np
from .utils import hann, sinc

_HAVE_SCIPY = True
try:
    from scipy.signal import firwin, filtfilt
except Exception:
    _HAVE_SCIPY = False

def fir_bandpass(numtaps, fs, f1, f2):
    M = int(numtaps)
    n = np.arange(M) - (M-1)/2.0
    h = (2*(f2/fs)*sinc(2*(f2/fs)*n) - 2*(f1/fs)*sinc(2*(f1/fs)*n))
    h *= hann(M); h /= np.sum(h) + 1e-12
    return h.astype(np.float32)

def design_bandpass(fs, f_lo, f_hi, taps):
    if _HAVE_SCIPY:
        return firwin(taps, [f_lo, f_hi], pass_zero=False, fs=fs).astype(np.float32)
    return fir_bandpass(taps, fs, f_lo, f_hi)

def bandpass_signal(x, bp):
    if _HAVE_SCIPY:
        return filtfilt(bp, [1.0], x).astype(np.float32)  # zero-phase (offline)
    y = np.convolve(x, bp, mode='full')
    gd = (len(bp)-1)//2
    return y[gd:gd+len(x)].astype(np.float32)
