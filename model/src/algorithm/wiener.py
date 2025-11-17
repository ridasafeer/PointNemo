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
    
    # Store original length for proper trimming later
    original_len = len(ambient)

    # ✅ Compute STFT
    f, t, A, backend = stft_any(ambient, fs=fs, nfft=nfft, hop=hop)
    f2, t2, X, _ = stft_any(x, fs=fs, nfft=nfft, hop=hop)
    
    print(f"DEBUG Wiener: backend={backend}, A.shape={A.shape}, A max={np.max(np.abs(A)):.6f}")

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
    print(f"DEBUG: S_xa max = {np.max(np.abs(S_xa)):.6f}, S_aa max = {np.max(np.abs(S_aa)):.6f}")

    # ✅ Synthesize anti-noise
    Yanti = Hf[:, None] * A
    print(f"DEBUG: Yanti max = {np.max(np.abs(Yanti)):.6f}")
    anti, backend_istft = istft_any(Yanti, fs=fs, nfft=nfft, hop=hop, backend=backend)
    print(f"DEBUG: anti (before compensation) max = {np.max(np.abs(anti)):.6f}, len={len(anti)}")
    
    # ✅ Overlap compensation - ALWAYS apply it (scipy istft with boundary=None doesn't do it correctly)
    anti = anti / (nfft / hop)
    
    # ✅ Trim to original length (critical!)
    anti = anti[:original_len]
    
    # ✅ Apply gain (NO CLIPPING!)
    anti = anti * gain
    
    print(f"DEBUG: anti max = {np.max(np.abs(anti)):.4f}, final len={len(anti)}")

    return anti[:len(x)].astype(np.float32), {'H_bins': Hf, 'lag': 0, 'gain': gain}