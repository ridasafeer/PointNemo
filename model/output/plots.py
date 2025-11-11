import os, numpy as np, matplotlib.pyplot as plt
# use an absolute import to reach into src/
from model.src.dsp.utils import welch_numpy


_HAVE_SCIPY = True
try:
    from scipy.signal import welch
except Exception:
    _HAVE_SCIPY = False

def plot_time(sig, fs, title, seconds=2.0, save=None):
    n = min(len(sig), int(seconds*fs))
    t = np.arange(n)/fs
    plt.figure(); plt.plot(t, sig[:n]); plt.xlabel('Time (s)'); plt.ylabel('Amp'); plt.title(title); plt.tight_layout()
    if save: plt.savefig(save, dpi=150); plt.close()
    else: plt.show()

def plot_psd(sig, fs, title, xlim=2000, save=None):
    if _HAVE_SCIPY: f, P = welch(sig, fs=fs, nperseg=4096)
    else:           f, P = welch_numpy(sig, fs, nperseg=4096, noverlap=2048)
    plt.figure(); plt.semilogy(f, P); plt.xlabel('Hz'); plt.ylabel('PSD'); plt.xlim(0, xlim); plt.title(title); plt.tight_layout()
    if save: plt.savefig(save, dpi=150); plt.close()
    else: plt.show()
    return f, P

def write_metrics_txt(path, *, in_name, fs, f_lo, f_hi, taps, solver, stats, p_before, p_after):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"Input: {in_name}, fs={fs}\n")
        fh.write(f"Band: {f_lo}-{f_hi} Hz, taps={taps}, solver={solver}, zero_phase=True\n")
        fh.write(f"Auto lag (samples): {int(stats.get('lag',0))}\n")
        fh.write(f"Gain (scalar or avg): {float(stats.get('gain',1.0)):.3f}\n")
        fh.write(f"Band power before: {p_before:.2f} dB\n")
        fh.write(f"Band power after : {p_after:.2f} dB\n")
        fh.write(f"Delta band power : {p_after - p_before:.2f} dB (negative is better)\n")