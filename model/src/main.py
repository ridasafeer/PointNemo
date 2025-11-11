# make repo root importable even when run by path
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]  # folder that contains 'model'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse, os
import numpy as np

from model.src.pn_io.wav import read_wav_mono, write_wav, resample_to
from model.src.dsp.filters import design_bandpass, bandpass_signal
from model.src.dsp.utils import ensure_dir, band_power_db, welch_numpy
from model.output.plots import plot_time, plot_psd, write_metrics_txt
from model.src.algorithm.scalar import solver_scalar
from model.src.algorithm.wiener import solver_wiener

_HAVE_SCIPY = True
try:
    from scipy.signal import welch
except Exception:
    _HAVE_SCIPY = False

def synthesize_hvac(fs=16000, duration=8.0, mains=60.0, seed=42):
    t = np.arange(int(fs*duration)) / fs
    hum = 0.15*np.sin(2*np.pi*mains*t) + 0.08*np.sin(2*np.pi*2*mains*t) + 0.05*np.sin(2*np.pi*3*mains*t)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(len(t)).astype(np.float32)
    ma = np.ones(401)/401
    lf = np.convolve(white, ma, mode='same')
    lf = lf - np.mean(lf); lf = lf / (np.max(np.abs(lf)) + 1e-8) * 0.2
    hiss = np.convolve(rng.standard_normal(len(t)).astype(np.float32), np.ones(33)/33, mode='same') * 0.03
    return (hum + lf + hiss).astype(np.float32)

def norm_peak(sig, eps=1e-9):
    pk = float(np.max(np.abs(sig)))
    if pk < eps:
        return np.zeros_like(sig, dtype=np.float32)
    return (0.9 / pk) * sig.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(description='PointNemo Offline ANC (modular)')
    ap.add_argument('--in', dest='in_wav', type=str, default=None)
    ap.add_argument('--fs', type=int, default=16000)
    ap.add_argument('--synthesize', action='store_true')
    ap.add_argument('--dur', type=float, default=8.0)
    ap.add_argument('--band', type=float, nargs=2, default=(20.0, 350.0))
    ap.add_argument('--taps', type=int, default=601)
    ap.add_argument('--anc_delay_ms', type=float, default=0.0)
    ap.add_argument('--solver', choices=['scalar','wiener'], default='wiener')
    ap.add_argument('--auto_align', action='store_true')
    ap.add_argument('--calib', type=float, default=2.0)
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--outdir', type=str, default='plots')
    ap.add_argument('--wavdir', type=str, default=os.path.join('model','output_auido'),
                help='Directory to write WAVs and metrics.txt')
    args = ap.parse_args()

    fs = args.fs
    f_lo, f_hi = args.band
    wavdir = args.wavdir
    ensure_dir(wavdir)

    # Load/synthesize
    if args.synthesize or args.in_wav is None:
        x = synthesize_hvac(fs=fs, duration=args.dur)
        in_name = f'synth_{args.dur:.1f}s'
    else:
        fs_in, x_in = read_wav_mono(args.in_wav)
        x = resample_to(x_in, fs_in, fs) if fs_in != fs else x_in.astype(np.float32)
        in_name = os.path.basename(args.in_wav)

    # Band-pass
    bp = design_bandpass(fs, f_lo, f_hi, args.taps)
    ambient = bandpass_signal(x, bp)

    # Solver
    if args.solver == 'scalar':
        anti, stats = solver_scalar(
            x, ambient, fs,
            calib_sec=args.calib, auto_align=args.auto_align, maxlag_ms=10.0
        )
    else:
        anti, stats = solver_wiener(
            x, ambient, fs, (f_lo, f_hi),
            nfft=4096, hop=None, reg=1e-4
        )

    # Optional extra anti delay
    if args.anc_delay_ms > 0:
        d = int(round(args.anc_delay_ms * 1e-3 * fs))
        anti = np.pad(anti, (d, 0))  # delay by padding at the start
        stats['lag'] = stats.get('lag', 0) + d

    # --- Make sure anti matches x length (fixes broadcasting issues) ---
    L = len(x)
    if len(anti) < L:
        anti = np.pad(anti, (0, L - len(anti)))
    elif len(anti) > L:
        anti = anti[:L]
    # -------------------------------------------------------------------

    residual = (x + anti).astype(np.float32)

    # Per-file normalization for audibility
    x_n, amb_n, anti_n, y_n = map(norm_peak, [x, ambient, anti, residual])

    # Save WAVs
    write_wav(os.path.join(wavdir, 'hvac_original.wav'), fs, x_n)
    write_wav(os.path.join(wavdir, f'hvac_ambient_band_{int(f_lo)}_{int(f_hi)}.wav'), fs, amb_n)
    write_wav(os.path.join(wavdir, 'hvac_anti_noise.wav'), fs, anti_n)
    write_wav(os.path.join(wavdir, 'hvac_residual_after_anc.wav'), fs, y_n)

    # Plots
    ensure_dir(args.outdir)
    if args.plots:
        plot_time(x_n,   fs, 'Original (first 2 s)',    save=os.path.join(args.outdir, '01_original_time.png'))
        fX, PX = plot_psd(x_n,   fs, 'Original - PSD',  save=os.path.join(args.outdir, '01_original_psd.png'))
        plot_time(amb_n, fs, 'Band-passed (first 2 s)', save=os.path.join(args.outdir, '02_band_time.png'))
        plot_psd(amb_n,  fs, 'Band-passed - PSD',       save=os.path.join(args.outdir, '02_band_psd.png'))
        plot_time(anti_n, fs, 'Anti-noise (first 2 s)', save=os.path.join(args.outdir, '03_anti_time.png'))
        plot_psd(anti_n,  fs, 'Anti-noise - PSD',       save=os.path.join(args.outdir, '03_anti_psd.png'))
        plot_time(y_n,    fs, 'Residual (first 2 s)',   save=os.path.join(args.outdir, '04_residual_time.png'))
        fY, PY = plot_psd(y_n,  fs, 'Residual - PSD',   save=os.path.join(args.outdir, '04_residual_psd.png'))

    # Metrics on RAW (un-normalized) signals
    if _HAVE_SCIPY:
        f0, P0 = welch(x,        fs=fs, nperseg=4096)
        f1, P1 = welch(residual, fs=fs, nperseg=4096)
    else:
        f0, P0 = welch_numpy(x,        fs, nperseg=4096, noverlap=2048)
        f1, P1 = welch_numpy(residual, fs, nperseg=4096, noverlap=2048)

    p_before = band_power_db(f0, P0, f_lo, f_hi)
    p_after  = band_power_db(f1, P1, f_lo, f_hi)

    write_metrics_txt(
    os.path.join(wavdir, 'metrics.txt'),
    in_name=in_name, fs=fs, f_lo=f_lo, f_hi=f_hi,
    taps=args.taps, solver=args.solver, stats=stats,
    p_before=p_before, p_after=p_after
    )

    print(f"Saved WAVs + metrics.txt. Δ band power = {(p_after - p_before):.2f} dB. Solver={args.solver}.")

if __name__ == '__main__':
    main()
