# model/src/algorithm/wiener.py
import numpy as np
from ..dsp.stft import stft_any, istft_any

def solver_wiener(x, ambient, fs, band, nfft=4096, hop=None, reg=1e-5, gain=1.0):
    """
    Compute H(f) = -S_xa / (S_aa + reg) and synthesize anti-noise.
    
    Parameters:
    -----------
    x : input reference signal
    ambient : ambient signal to cancel
    fs : sample rate
    band : (lo, hi) frequency band
    nfft : FFT size
    reg : regularization (lower = more aggressive)
    gain : output scaling factor
    """
    if hop is None:
        hop = nfft // 2

    # ✅ Compute STFT
    f, t, A = stft_any(ambient, fs=fs, nfft=nfft, hop=hop)
    f2, t2, X = stft_any(x, fs=fs, nfft=nfft, hop=hop)

    # ✅ Cross/auto spectra
    S_xa = np.sum(X * np.conj(A), axis=1)
    S_aa = np.sum(A * np.conj(A), axis=1) + reg

    # ✅ Wiener filter
    Hf = -S_xa / S_aa

    # ✅ Limit to band
    lo, hi = band
    mask = (f >= lo) & (f <= hi)
    Hf = Hf * mask

    print(f"DEBUG: reg={reg}, gain={gain}")
    print(f"DEBUG: Hf max = {np.max(np.abs(Hf)):.4f}")

    # ✅ Synthesize anti-noise
    Yanti = Hf[:, None] * A
    anti = istft_any(Yanti, fs=fs, nfft=nfft, hop=hop)
    
    # ✅ Overlap compensation
    anti = anti / (nfft / hop)
    
    # ✅ Apply gain (NO CLIPPING!)
    anti = anti * gain
    
    print(f"DEBUG: anti max = {np.max(np.abs(anti)):.4f}")

    return anti[:len(x)].astype(np.float32), {'H_bins': Hf, 'lag': 0, 'gain': gain}