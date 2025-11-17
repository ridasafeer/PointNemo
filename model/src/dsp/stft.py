import numpy as np
_HAVE_SCIPY = True  # Scipy works correctly now
try:
    from scipy.signal import stft as _stft, istft as _istft, get_window
except Exception:
    _HAVE_SCIPY = False

def stft_any(x, fs, nfft=4096, hop=None):
    if hop is None:
        hop = nfft // 2
    if _HAVE_SCIPY:
        print(f"DEBUG STFT: Using scipy")
        win = get_window('hann', nfft, fftbins=False)
        # Use boundary=None to match numpy's frame count
        f, t, X = _stft(
            x, fs=fs,
            window=win,
            nperseg=nfft,
            noverlap=nfft - hop,
            boundary=None,
            padded=False,
            return_onesided=True
        )
        # Normalize scipy to match numpy's scaling
        win_sum = np.sum(win)
        win_norm = np.sum(win**2)
        scale_factor = win_sum / np.sqrt(win_norm)
        X = X * scale_factor
        
        print(f"DEBUG STFT: X max = {np.max(np.abs(X))}, X shape = {X.shape}")
        return f, t, X, 'scipy'  # Return flag
    else:
        print(f"DEBUG STFT: Using NumPy fallback")
        win = np.hanning(nfft)
        win_norm = np.sum(win**2)  # ✅ For proper COLA normalization
        frames = []
        for i in range(0, len(x) - nfft + 1, hop):
            frame = np.fft.rfft(x[i:i+nfft] * win)
            frame = frame / np.sqrt(win_norm)  # ✅ Normalize by window energy
            frames.append(frame)
        f = np.fft.rfftfreq(nfft, d=1/fs)
        t = np.arange(len(frames)) * hop / fs
        print(f"DEBUG STFT NumPy: X max = {np.max(np.abs(np.array(frames)))}, X shape = {np.array(frames).T.shape}")
        return f, t, np.array(frames).T, 'numpy'  # Return flag

def istft_any(X, fs, nfft=4096, hop=None, backend='scipy'):
    if hop is None:
        hop = nfft // 2
    if _HAVE_SCIPY and backend == 'scipy':
        # Scipy's istft with boundary=None produces incorrect length
        # Use numpy's implementation instead for consistency
        backend = 'numpy'
    
    # ✅ NumPy overlap-add with PROPER normalization
    win = np.hanning(nfft)
    win_norm = np.sum(win**2)
    H = hop
    T = X.shape[1]
    out_len = H * (T - 1) + nfft
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    
    for i in range(T):
        frame = np.fft.irfft(X[:, i], n=nfft)
        frame = frame * win / np.sqrt(win_norm)  # ✅ Apply window with normalization
        s = i * H
        y[s:s+nfft] += frame
        wsum[s:s+nfft] += win**2 / win_norm  # ✅ Normalized window sum
    
    # ✅ Avoid division by zero
    wsum[wsum < 1e-8] = 1.0
    
    return (y / wsum).astype(np.float32), 'numpy'  # Always return numpy since we use numpy impl
    
    # ✅ NumPy overlap-add with PROPER normalization
    win = np.hanning(nfft)
    win_norm = np.sum(win**2)
    H = hop
    T = X.shape[1]
    out_len = H * (T - 1) + nfft
    y = np.zeros(out_len)
    wsum = np.zeros(out_len)
    
    for i in range(T):
        frame = np.fft.irfft(X[:, i], n=nfft)
        frame = frame * win / np.sqrt(win_norm)  # ✅ Apply window with normalization
        s = i * H
        y[s:s+nfft] += frame
        wsum[s:s+nfft] += win**2 / win_norm  # ✅ Normalized window sum
    
    # ✅ Avoid division by zero
    wsum[wsum < 1e-8] = 1.0
    
    return (y / wsum).astype(np.float32), 'numpy'  # Return flag
