# model/src/algorithm/wiener.py
import numpy as np
from ..dsp.stft import stft_any, istft_any  # relative import into dsp

def solver_wiener(x, ambient, fs, band, nfft=4096, hop=None, reg=1e-4):
    """
    Compute H(f) = -S_xa / (S_aa + reg) and synthesize anti = iSTFT{ H(f)*A(f) }.
    'band' is a (lo, hi) tuple in Hz; H is limited to that band for stability.
    """
    if hop is None:
        hop = nfft // 2

    f, t, A = stft_any(ambient, fs=fs, nfft=nfft, hop=hop)
    f2, t2, X = stft_any(x,       fs=fs, nfft=nfft, hop=hop)

    # Cross/auto spectra (sum over time frames)
    S_xa = np.sum(X * np.conj(A), axis=1)
    S_aa = np.sum(A * np.conj(A), axis=1) + reg

    # Per-bin complex mapping from ambient->(-x)
    Hf = - S_xa / S_aa

    # Limit to target band
    lo, hi = band
    mask = (f >= lo) & (f <= hi)
    Hf = Hf * mask

    # Synthesize anti-noise
    Yanti = Hf[:, None] * A
    anti = istft_any(Yanti, fs=fs, nfft=nfft, hop=hop)

    return anti[:len(x)].astype(np.float32), {'H_bins': Hf, 'lag': 0, 'gain': 1.0}
