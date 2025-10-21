#!/usr/bin/env python3
"""
anc_sim_new.py - PointNemo ANC (drop-in)
----------------------------------------
Self-contained ANC simulation with causal streaming, alignment, optional
hum notches, and auto-gain calibration. Designed to run from anywhere
(no imports from your lab's model package).

USAGE (Windows PowerShell examples)
  # Use your full HVAC recording (recommended), not the already-bandpassed file
  python .\anc_sim_new.py --in ..\hvac_original.wav --band 20 500 --L 256 --taps 801 --plots --outdir plots

  # Add hum notches (NOTE: notches here suppress those tones before anti-noise)
  python .\anc_sim_new.py --in ..\hvac_original.wav --band 20 500 --L 256 --taps 801 --notch 60 120 180 --plots

  # Upper-bound / offline check (zero-phase)
  python .\anc_sim_new.py --in ..\hvac_original.wav --band 20 500 --zero_phase --plots

  # Run built-in self tests
  python .\anc_sim_new.py --selftest

OUTPUTS
  hvac_original.wav
  hvac_ambient_band_<lo>_<hi>.wav
  hvac_anti_noise.wav
  hvac_residual_after_anc.wav
  metrics.txt (band power before/after, delta dB, RMS, calibrated gain, group delay)
  plots/ (if --plots): time/PSD/spectrogram images for each stage

NOTE
 • If you feed it the already-filtered file (hvac_ambient_band_20_500.wav),
   you will double-filter the band and the result may sound odd. Use your
   full/unaltered recording for realistic listening tests.
"""

import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# Try SciPy; if missing, raise a clear error
try:
    from scipy.io import wavfile
    from scipy import signal
    HAVE_SCIPY = True
except Exception as e:
    HAVE_SCIPY = False
    raise SystemExit("This script requires SciPy (scipy.io.wavfile, scipy.signal). Install with: pip install scipy")

# ---------------------------- IO helpers ----------------------------

def read_wav_mono(path):
    fs, x = wavfile.read(path)
    if x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        y = (x.astype(np.float32) - 128) / 128.0
    else:
        y = x.astype(np.float32)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return fs, y

def write_wav(path, fs, x):
    x = np.clip(x, -1.0, 1.0)
    wavfile.write(path, fs, (x * 32767.0).astype(np.int16))

# ---------------------------- DSP helpers ---------------------------

def welch_psd(x, fs, nperseg=4096):
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg)
    return f, Pxx

def band_power_db(f, Pxx, lo, hi):
    idx = (f >= lo) & (f <= hi)
    if not np.any(idx):
        return float('nan')
    df = f[1] - f[0]
    return 10*np.log10(np.sum(Pxx[idx]) * df + 1e-20)

def design_bandpass_fir(fs, f_lo, f_hi, taps):
    return signal.firwin(taps, [f_lo, f_hi], pass_zero=False, fs=fs, window='hann').astype(np.float32)

def design_notches_ba(fs, tones, Q=30):
    """Return a list of (b, a) notch sections. Compatible with older SciPy.
    Tries the Hz API first and falls back to normalized frequency.
    """
    sections = []
    for f0 in tones:
        if f0 <= 0 or f0 >= fs/2:
            continue
        # Try modern API (Hz)
        try:
            b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
        except TypeError:
            # Older SciPy: w0 is normalized (0..1)
            w0 = f0/(fs/2.0)
            b, a = signal.iirnotch(w0, Q)
        sections.append((b.astype(np.float64), a.astype(np.float64)))
    return sections

def apply_notches(x, sections, zero_phase=False):
    y = x
    for (b, a) in sections:
        if zero_phase:
            try:
                y = signal.filtfilt(b, a, y)
            except Exception:
                y = signal.lfilter(b, a, y)
        else:
            y = signal.lfilter(b, a, y)
    return y

# ------------------------- Plotting helpers -------------------------

def ensure_dir(path):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def save_time_plot(sig, fs, outpath, title, seconds=2.0):
    n = min(len(sig), int(seconds*fs))
    t = np.arange(n)/fs
    plt.figure()
    plt.plot(t, sig[:n])
    plt.xlabel('Time (s)'); plt.ylabel('Amplitude'); plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_psd_plot(sig, fs, outpath, title, xlim=2000):
    f, P = welch_psd(sig, fs)
    plt.figure()
    plt.semilogy(f, P)
    plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD'); plt.xlim(0, xlim); plt.title(title)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_specgram(sig, fs, outpath, title):
    plt.figure()
    plt.specgram(sig, NFFT=1024, Fs=fs, noverlap=512)
    plt.xlabel('Time (s)'); plt.ylabel('Freq (Hz)'); plt.title(title)
    plt.colorbar(label='dB'); plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# ---------------------------- ANC core ------------------------------

def anc_stream(x, fs, f_lo, f_hi, taps=801, L=256, notches=(), zero_phase=False, calib_sec=2.0):
    """Run ANC pipeline and return (ambient, anti, residual, stats).
    stats contains: gain, group_delay_samples, and scaled intermediates for plotting.
    """
    x = x.astype(np.float32)

    # Bandpass FIR & group delay
    h = design_bandpass_fir(fs, f_lo, f_hi, taps)
    gd = (taps - 1)//2

    # Optional notches (list of (b,a) sections)
    notch_sections = design_notches_ba(fs, notches) if notches else []

    # --- Stage A: Bandpass only ---
    if zero_phase:
        ambient_bp = signal.filtfilt(h, [1.0], x).astype(np.float32)
    else:
        # Causal streaming with lfilter (block by block)
        zi_band = signal.lfilter_zi(h, 1.0) * 0.0
        out_bp = []
        i = 0
        while i < len(x):
            blk = x[i:i+L]
            yb, zi_band = signal.lfilter(h, [1.0], blk, zi=zi_band)
            out_bp.append(yb.astype(np.float32))
            i += L
        ambient_bp = np.concatenate(out_bp)

    # --- Stage B: Apply notches (optional) ---
    if notch_sections:
        ambient = apply_notches(ambient_bp, notch_sections, zero_phase=zero_phase).astype(np.float32)
    else:
        ambient = ambient_bp

    # --- Stage C: Anti-noise before and after gain ---
    anti_pre_gain = -ambient

    # Alignment: delay direct path in streaming mode
    if zero_phase:
        dpath = x
        align_lag = 0
    else:
        dpath = np.pad(x, (gd, 0))[:len(x)]
        align_lag = gd

    # Calibrate global gain on first calib_sec seconds
    Ncal = min(len(x), int(calib_sec*fs))
    if Ncal > taps:
        t = -ambient
        g = float(-np.dot(dpath[:Ncal], t[:Ncal]) / (np.dot(t[:Ncal], t[:Ncal]) + 1e-20))
    else:
        g = 1.0

    anti = anti_pre_gain * g
    residual = dpath + anti

    # Normalize for export/plots (common scale)
    peak = max(1e-6, np.max(np.abs(np.concatenate([x, ambient_bp, ambient, anti, residual, dpath]))))
    s = 0.95/peak

    stats = {
        "gain": g,
        "group_delay_samples": gd,
        # Scaled intermediates for plotting
        "x_s": x*s,
        "x_delayed_s": dpath*s,
        "ambient_bp_s": ambient_bp*s,
        "ambient_post_s": ambient*s,
        "anti_pre_gain_s": anti_pre_gain*s,
        "anti_s": anti*s,
        "residual_s": residual*s,
    }

    return ambient*s, anti*s, residual*s, stats

# ------------------------------- CLI --------------------------------

def run_self_tests():
    """Basic self-tests to catch regressions.
    These run quickly and do not need any external files.
    """
    fs = 16000
    t = np.arange(int(fs*6.0))/fs
    # Synthesize HVAC-like signal: hum + rumble + hiss
    hum = 0.15*np.sin(2*np.pi*60*t) + 0.08*np.sin(2*np.pi*120*t) + 0.05*np.sin(2*np.pi*180*t)
    rng = np.random.default_rng(7)
    white = rng.standard_normal(len(t)).astype(np.float32)
    h_lp = signal.firwin(801, [20, 300], pass_zero=False, fs=fs)
    rumble = signal.filtfilt(h_lp, [1.0], white)
    rumble *= 0.2/ (np.max(np.abs(rumble))+1e-9)
    hiss = signal.filtfilt(signal.firwin(801, [1500, 4000], pass_zero=False, fs=fs), [1.0], white) * 0.03
    x = (hum + rumble + hiss).astype(np.float32)

    # Helper to measure band delta
    def delta_band(x_in, x_out, lo=20.0, hi=500.0):
        f0, P0 = welch_psd(x_in, fs)
        f1, P1 = welch_psd(x_out, fs)
        return band_power_db(f1, P1, lo, hi) - band_power_db(f0, P0, lo, hi)

    # Test 1: zero-phase should reduce LF by at least ~8 dB
    amb, anti, res, st = anc_stream(x, fs, 20.0, 500.0, taps=801, L=256, notches=(), zero_phase=True)
    d1 = delta_band(x, res)
    assert d1 < -6.0, f"Zero-phase attenuation too small: {d1:.2f} dB"

    # Test 2: streaming should reduce LF by at least ~3 dB
    amb, anti, res, st = anc_stream(x, fs, 20.0, 500.0, taps=801, L=256, notches=(), zero_phase=False)
    d2 = delta_band(x, res)
    assert d2 < -3.0, f"Streaming attenuation too small: {d2:.2f} dB"

    # Test 3: notch path runs without NameError and returns arrays of same length
    amb, anti, res, st = anc_stream(x, fs, 20.0, 500.0, taps=801, L=256, notches=(60.0, 120.0), zero_phase=False)
    assert len(amb) == len(anti) == len(res) == len(x), "Output lengths mismatch with notches"

    # Test 4: stages exist in stats for plotting
    for k in ["x_s","x_delayed_s","ambient_bp_s","ambient_post_s","anti_pre_gain_s","anti_s","residual_s"]:
        assert k in st and isinstance(st[k], np.ndarray), f"Missing stage {k} in stats"

    print("Self-tests passed:")
    print(f" - Zero-phase delta dB: {d1:.2f}")
    print(f" - Streaming  delta dB: {d2:.2f}")


def main():
    ap = argparse.ArgumentParser(description='PointNemo ANC (drop-in)')
    g_in = ap.add_mutually_exclusive_group(required=False)
    g_in.add_argument('--in', dest='in_wav', type=str, help='Input HVAC WAV (mono/stereo)')
    g_in.add_argument('--synthesize', action='store_true', help='Use synthesized HVAC noise')
    ap.add_argument('--fs', type=int, default=16000, help='Processing sample rate')
    ap.add_argument('--dur', type=float, default=12.0, help='Duration when synthesizing (s)')
    ap.add_argument('--band', nargs=2, type=float, default=(20.0, 500.0), help='Band to cancel [lo hi] Hz')
    ap.add_argument('--taps', type=int, default=801, help='FIR taps (odd recommended)')
    ap.add_argument('--L', type=int, default=256, help='Block length for streaming (samples)')
    ap.add_argument('--notch', nargs='*', type=float, default=[], help='Hum notches in Hz, e.g., 60 120 180')
    ap.add_argument('--zero_phase', action='store_true', help='Use zero-phase (offline) mode')
    ap.add_argument('--calib', type=float, default=2.0, help='Calibration window (s) for gain')
    ap.add_argument('--plots', action='store_true', help='Save plots for all stages to --outdir')
    ap.add_argument('--outdir', type=str, default='plots', help='Directory to save plots')
    ap.add_argument('--selftest', action='store_true', help='Run built-in tests and exit')
    args = ap.parse_args()

    if args.selftest:
        run_self_tests()
        return

    fs = args.fs

    # Load or synthesize
    if args.synthesize or not args.in_wav:
        t = np.arange(int(fs*args.dur))/fs
        hum = 0.15*np.sin(2*np.pi*60*t) + 0.08*np.sin(2*np.pi*120*t) + 0.05*np.sin(2*np.pi*180*t)
        rng = np.random.default_rng(7)
        white = rng.standard_normal(len(t)).astype(np.float32)
        h_lp = signal.firwin(801, [20, 300], pass_zero=False, fs=fs)
        rumble = signal.filtfilt(h_lp, [1.0], white)
        rumble *= 0.2/ (np.max(np.abs(rumble))+1e-9)
        hiss = signal.filtfilt(signal.firwin(801, [1500, 4000], pass_zero=False, fs=fs), [1.0], white) * 0.03
        x = (hum + rumble + hiss).astype(np.float32)
        in_name = 'synth'
    else:
        fs_in, xin = read_wav_mono(args.in_wav)
        if fs_in != fs:
            dur = len(xin)/fs_in
            t_old = np.linspace(0, dur, len(xin), endpoint=False)
            t_new = np.linspace(0, dur, int(dur*fs), endpoint=False)
            x = np.interp(t_new, t_old, xin).astype(np.float32)
        else:
            x = xin.astype(np.float32)
        in_name = os.path.basename(args.in_wav)

    f_lo, f_hi = args.band
    ambient, anti, residual, stats = anc_stream(
        x, fs, f_lo, f_hi, taps=args.taps, L=args.L, notches=args.notch,
        zero_phase=args.zero_phase, calib_sec=args.calib,
    )

    # Save WAVs (scaled)
    write_wav('hvac_original.wav', fs, stats['x_s'])
    write_wav(f'hvac_ambient_band_{int(f_lo)}_{int(f_hi)}.wav', fs, ambient)
    write_wav('hvac_anti_noise.wav', fs, anti)
    write_wav('hvac_residual_after_anc.wav', fs, residual)

    # Metrics
    f0, P0 = welch_psd(stats['x_s'], fs)
    f1, P1 = welch_psd(residual, fs)
    p_before = band_power_db(f0, P0, f_lo, f_hi)
    p_after  = band_power_db(f1, P1, f_lo, f_hi)
    delta = p_after - p_before
    rms = lambda s: 20*np.log10(np.sqrt(np.mean(s*s)+1e-20)+1e-20)

    with open('metrics.txt', 'w', encoding='utf-8') as fh:
        fh.write(f"Input: {in_name}, fs={fs}\n")
        fh.write(f"Band: {f_lo}-{f_hi} Hz, taps={args.taps}, L={args.L}, notches={args.notch}, zero_phase={args.zero_phase}\n")
        fh.write(f"Group delay (samples): {stats['group_delay_samples']}\n")
        fh.write(f"Calibrated anti-noise gain: {stats['gain']:.3f}\n")
        fh.write(f"Band power before: {p_before:.2f} dB\n")
        fh.write(f"Band power after : {p_after:.2f} dB\n")
        fh.write(f"Delta band power      : {delta:.2f} dB (negative is better)\n")
        fh.write(f"RMS original (dBFS): {rms(stats['x_s']):.2f}\n")
        fh.write(f"RMS residual (dBFS): {rms(residual):.2f}\n")

    print(f"Saved WAVs + metrics.txt. Delta band power = {delta:.2f} dB. Gain={stats['gain']:.3f}, GD={stats['group_delay_samples']} samples")

    # Save plots per stage if requested
    if args.plots:
        outdir = args.outdir or 'plots'
        ensure_dir(outdir)
        # Stage 1: Original
        save_time_plot(stats['x_s'], fs, os.path.join(outdir, 'stage1_original_time.png'), 'Original (first 2 s)')
        save_psd_plot(stats['x_s'], fs, os.path.join(outdir, 'stage1_original_psd.png'), 'Original - PSD')
        save_specgram(stats['x_s'], fs, os.path.join(outdir, 'stage1_original_spec.png'), 'Original - Spectrogram')
        # Stage 2: Bandpass output
        save_time_plot(stats['ambient_bp_s'], fs, os.path.join(outdir, 'stage2_bandpass_time.png'), 'Bandpass (first 2 s)')
        save_psd_plot(stats['ambient_bp_s'], fs, os.path.join(outdir, 'stage2_bandpass_psd.png'), 'Bandpass - PSD')
        # Stage 3: After notches (if applied)
        save_time_plot(stats['ambient_post_s'], fs, os.path.join(outdir, 'stage3_post_notch_time.png'), 'Post-notch (first 2 s)')
        save_psd_plot(stats['ambient_post_s'], fs, os.path.join(outdir, 'stage3_post_notch_psd.png'), 'Post-notch - PSD')
        # Stage 4: Anti-noise pre-gain
        save_time_plot(stats['anti_pre_gain_s'], fs, os.path.join(outdir, 'stage4_anti_pre_gain_time.png'), 'Anti-noise (pre-gain) - first 2 s')
        save_psd_plot(stats['anti_pre_gain_s'], fs, os.path.join(outdir, 'stage4_anti_pre_gain_psd.png'), 'Anti-noise (pre-gain) - PSD')
        # Stage 5: Anti-noise post-gain
        save_time_plot(stats['anti_s'], fs, os.path.join(outdir, 'stage5_anti_post_gain_time.png'), 'Anti-noise (post-gain) - first 2 s')
        save_psd_plot(stats['anti_s'], fs, os.path.join(outdir, 'stage5_anti_post_gain_psd.png'), 'Anti-noise (post-gain) - PSD')
        # Stage 6: Residual
        save_time_plot(stats['residual_s'], fs, os.path.join(outdir, 'stage6_residual_time.png'), 'Residual (first 2 s)')
        save_psd_plot(stats['residual_s'], fs, os.path.join(outdir, 'stage6_residual_psd.png'), 'Residual - PSD')
        save_specgram(stats['residual_s'], fs, os.path.join(outdir, 'stage6_residual_spec.png'), 'Residual - Spectrogram')
        # Overlay comparison: Original vs Residual PSD
        f0, P0 = welch_psd(stats['x_s'], fs)
        f1, P1 = welch_psd(stats['residual_s'], fs)
        plt.figure(); plt.semilogy(f0, P0, label='Original'); plt.semilogy(f1, P1, label='Residual');
        plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD'); plt.xlim(0, 2000); plt.legend(); plt.title('PSD: Original vs Residual'); plt.tight_layout();
        plt.savefig(os.path.join(outdir, 'overlay_psd_original_vs_residual.png'), dpi=150); plt.close()

if __name__ == '__main__':
    main()
