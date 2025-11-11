import numpy as np
_HAVE_SCIPY = True
try:
    from scipy.signal import stft as _stft, istft as _istft, get_window
except Exception:
    _HAVE_SCIPY = False

def stft_any(x, fs, nfft=4096, hop=None):
    if hop is None:
        hop = nfft // 2  # 50% overlap for Hann -> COLA/NOLA satisfied
    if _HAVE_SCIPY:
        win = get_window('hann', nfft, fftbins=True)
        f, t, X = _stft(
            x, fs=fs,
            window=win,
            nperseg=nfft,
            noverlap=nfft - hop,
            boundary=None,   # avoid zero-padding that breaks NOLA check
            padded=False,
            return_onesided=True
        )
        return f, t, X
    # NumPy fallback
    win = np.hanning(nfft); H = hop
    frames = [np.fft.rfft(x[i:i+nfft] * win) for i in range(0, len(x) - nfft + 1, H)]
    f = np.fft.rfftfreq(nfft, d=1/fs); t = np.arange(len(frames)) * H / fs
    return f, t, np.array(frames).T

def istft_any(X, fs, nfft=4096, hop=None):
    if hop is None:
        hop = nfft // 2
    if _HAVE_SCIPY:
        win = get_window('hann', nfft, fftbins=True)
        _, y = _istft(
            X, fs=fs,
            window=win,
            nperseg=nfft,
            noverlap=nfft - hop,
            input_onesided=True
        )
        return y.astype(np.float32)
    # NumPy overlap-add
    win = np.hanning(nfft); H = hop
    T = X.shape[1]; out_len = H * (T - 1) + nfft
    y = np.zeros(out_len); wsum = np.zeros(out_len)
    for i in range(T):
        frame = np.fft.irfft(X[:, i])
        s = i * H
        y[s:s+nfft] += frame * win
        wsum[s:s+nfft] += win**2
    wsum[wsum==0] = 1.0
    return (y/wsum).astype(np.float32)
