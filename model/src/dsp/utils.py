import os, numpy as np

def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def hann(N):
    n = np.arange(N)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(N-1))

def sinc(x):
    y = np.ones_like(x); nz = x != 0
    y[nz] = np.sin(np.pi*x[nz])/(np.pi*x[nz])
    return y

def welch_numpy(x, fs, nperseg=4096, noverlap=2048):
    step = nperseg - noverlap if nperseg - noverlap > 0 else nperseg//2
    win = np.hanning(nperseg); scale = (fs*(win**2).sum())
    acc = []
    for start in range(0, len(x)-nperseg+1, step):
        seg = x[start:start+nperseg]*win
        Pxx = (np.abs(np.fft.rfft(seg))**2)/scale
        acc.append(Pxx)
    if not acc:
        f = np.fft.rfftfreq(nperseg, d=1/fs)
        return f, np.zeros_like(f)
    P = np.mean(np.stack(acc,0),0)
    f = np.fft.rfftfreq(nperseg, d=1/fs)
    return f, P

def band_power_db(f, P, lo, hi):
    idx = (f>=lo) & (f<=hi)
    if not np.any(idx): return float('nan')
    df = f[1]-f[0]
    return 10*np.log10(np.sum(P[idx])*df + 1e-20)

def shift_signal(y, lag):
    if lag == 0: return y
    if lag > 0:  return np.pad(y, (lag,0))[:len(y)]
    return np.pad(y, (0,-lag))[(-lag):]
