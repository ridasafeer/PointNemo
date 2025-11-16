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

def plot_comparison(x_orig, y_resid, fs, title, save=None):
    """Plot original vs residual signals for direct ANC effectiveness comparison."""
    max_samples = int(2.0 * fs)  # 2 seconds
    x_plot = x_orig[:max_samples]
    y_plot = y_resid[:max_samples]
    t = np.arange(len(x_plot)) / fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Original signal
    ax1.plot(t, x_plot, linewidth=0.8, color='red', label='Original (Input)')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Original Signal (Before ANC)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Residual signal
    ax2.plot(t, y_plot, linewidth=0.8, color='green', label='Residual (After ANC)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Residual Signal (After ANC)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Match y-axis scales for comparison
    y_lim = max(np.max(np.abs(x_plot)), np.max(np.abs(y_plot))) * 1.1
    ax1.set_ylim([-y_lim, y_lim])
    ax2.set_ylim([-y_lim, y_lim])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_psd_comparison(f0, P0, f1, P1, f_lo, f_hi, save=None):
    """Plot PSD comparison showing cancellation in frequency domain."""
    P0_db = 10 * np.log10(P0 + 1e-12)
    P1_db = 10 * np.log10(P1 + 1e-12)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.semilogy(f0, P0, linewidth=1.2, label='Original (Before ANC)', color='red', alpha=0.8)
    ax.semilogy(f1, P1, linewidth=1.2, label='Residual (After ANC)', color='green', alpha=0.8)
    
    # Highlight the target band
    ax.axvspan(f_lo, f_hi, alpha=0.1, color='blue', label=f'Target Band ({f_lo}-{f_hi} Hz)')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (V²/Hz)', fontsize=11)
    ax.set_title('PSD Comparison: ANC Effectiveness', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([0, 500])  # Focus on low frequencies where most energy is
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_bandpass_vs_residual(amb, y_resid, fs, title, save=None):
    """Plot bandpass signal vs residual overlaid for direct cancellation comparison."""
    max_samples = int(2.0 * fs)  # 2 seconds
    amb_plot = amb[:max_samples]
    y_plot = y_resid[:max_samples]
    t = np.arange(len(amb_plot)) / fs
    
    # Find max across both signals for fair comparison
    max_amp = max(np.max(np.abs(amb_plot)), np.max(np.abs(y_plot)))
    y_lim = max_amp * 1.1
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both signals overlaid
    ax.plot(t, amb_plot, linewidth=1.0, label='Bandpass Signal (Input to ANC)', 
            color='blue', alpha=0.7)
    ax.plot(t, y_plot, linewidth=1.0, label='Residual (After ANC)', 
            color='green', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim([-y_lim, y_lim])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_anc_reduction(f0, P0, f1, P1, f_lo, f_hi, save=None):
    """Plot dB reduction achieved by ANC at each frequency."""
    
    P0_db = 10 * np.log10(P0 + 1e-12)
    P1_db = 10 * np.log10(P1 + 1e-12)
    
    # Calculate dB reduction (negative = cancellation)
    reduction_db = P1_db - P0_db
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot reduction
    ax.plot(f0, reduction_db, linewidth=1.5, color='darkgreen', label='ANC Reduction (dB)')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No cancellation')
    ax.axvspan(f_lo, f_hi, alpha=0.1, color='blue', label=f'Target Band ({f_lo}-{f_hi} Hz)')
    
    # Fill negative area (cancellation)
    ax.fill_between(f0, reduction_db, 0, where=(reduction_db < 0), alpha=0.3, color='green', label='Cancelled')
    ax.fill_between(f0, reduction_db, 0, where=(reduction_db >= 0), alpha=0.3, color='red', label='Amplified')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Reduction (dB)', fontsize=11)
    ax.set_title('ANC System Performance: dB Reduction by Frequency', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 500])
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_bandpass_vs_anti_noise(amb, anti, fs, title, save=None):
    """Plot bandpass signal vs anti-noise overlaid."""
    max_samples = int(2.0 * fs)  # 2 seconds
    amb_plot = amb[:max_samples]
    anti_plot = anti[:max_samples]
    t = np.arange(len(amb_plot)) / fs
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Bandpass signal
    ax1.plot(t, amb_plot, linewidth=0.8, color='blue', label='Bandpass Signal')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Bandpass Signal (20-350 Hz)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Anti-noise signal
    ax2.plot(t, anti_plot, linewidth=0.8, color='red', label='Anti-noise Signal')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title('Generated Anti-noise', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Match y-axis scales
    y_lim = max(np.max(np.abs(amb_plot)), np.max(np.abs(anti_plot))) * 1.1
    ax1.set_ylim([-y_lim, y_lim])
    ax2.set_ylim([-y_lim, y_lim])
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)

def plot_bandpass_anti_overlay(amb, anti, fs, title, save=None):
    """Plot bandpass and anti-noise on same plot (overlaid)."""
    max_samples = int(2.0 * fs)  # 2 seconds
    amb_plot = amb[:max_samples]
    anti_plot = anti[:max_samples]
    t = np.arange(len(amb_plot)) / fs
    
    # Match scales
    y_lim = max(np.max(np.abs(amb_plot)), np.max(np.abs(anti_plot))) * 1.1
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(t, amb_plot, linewidth=1.0, label='Bandpass Signal (Input)', 
            color='blue', alpha=0.7)
    ax.plot(t, anti_plot, linewidth=1.0, label='Anti-noise Signal (Generated)', 
            color='red', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim([-y_lim, y_lim])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=100, bbox_inches='tight')
        print(f"Saved: {save}")
    plt.close(fig)