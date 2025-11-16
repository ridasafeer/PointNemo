import numpy as np
import matplotlib.pyplot as plt
import os

_HAVE_SCIPY = True
try:
    from scipy.signal import welch
except Exception:
    _HAVE_SCIPY = False

from model.src.dsp.utils import welch_numpy

def plot_time(x, fs, title, save=None):
    """Plot first 2 seconds of signal in time domain."""
    max_samples = int(2.0 * fs)  # 2 seconds
    x_plot = x[:max_samples]
    t = np.arange(len(x_plot)) / fs
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, x_plot, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_psd(x, fs, title, save=None):
    """Plot Power Spectral Density using Welch method."""
    if _HAVE_SCIPY:
        f, P = welch(x, fs=fs, nperseg=4096)
    else:
        f, P = welch_numpy(x, fs, nperseg=4096, noverlap=2048)
    
    # Convert to dB
    P_db = 10 * np.log10(P + 1e-12)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogy(f, P, linewidth=1.0)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0, fs / 2])
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)
    
    return f, P

def write_metrics_txt(filepath, in_name, fs, f_lo, f_hi, taps, solver, stats, p_before, p_after):
    """Write metrics to text file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PointNemo ANC Metrics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("INPUT\n")
        f.write(f"  File: {in_name}\n")
        f.write(f"  Sampling Rate: {fs} Hz\n\n")
        
        f.write("FILTERING\n")
        f.write(f"  Band: {f_lo} - {f_hi} Hz\n")
        f.write(f"  FIR Taps: {taps}\n\n")
        
        f.write("SOLVER\n")
        f.write(f"  Algorithm: {solver}\n")
        if 'gain' in stats:
            f.write(f"  Gain: {stats['gain']:.4f}\n")
        if 'lag' in stats:
            f.write(f"  Lag (samples): {stats['lag']}\n\n")
        
        f.write("PERFORMANCE\n")
        f.write(f"  Power Before ANC: {p_before:.2f} dB\n")
        f.write(f"  Power After ANC:  {p_after:.2f} dB\n")
        f.write(f"  Delta Band Power: {(p_after - p_before):.2f} dB\n")
        f.write("=" * 60 + "\n")
    
    print(f"Saved: {filepath}")