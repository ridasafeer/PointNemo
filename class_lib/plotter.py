import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, spectrogram, find_peaks, get_window

class WavAnalysis:
    def __init__(self, path):
        self.fs, data = wavfile.read(path)
        if data.ndim > 1:
            data = data[:, 0]
        # float, mono, DC-removed, normalized
        x = data.astype(np.float64)
        x -= x.mean()
        x /= np.max(np.abs(x)) + 1e-12
        self.x = x
        self.N = len(x)

    # 1) Power spectrum (Welch). Good for “what frequencies are highest?”
    def plot_power_spectrum(self, nperseg=4096, overlap=0.5, window='hann'):
        noverlap = int(nperseg * overlap)
        f, Pxx = welch(self.x, fs=self.fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, detrend='constant', return_onesided=True)
        Pxx_db = 10 * np.log10(Pxx + 1e-20)
        plt.figure()
        plt.semilogx(f, Pxx_db)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title("Power spectrum (Welch)")
        plt.grid(True, which='both')
        plt.show()

    # 2) List top-k frequency peaks from the power spectrum with human-readable output
    def top_frequencies(self, k=5, nperseg=4096, overlap=0.5, window='hann',
                        min_hz=20, max_hz=None, as_text=True, show=True):
        noverlap = int(nperseg * overlap)
        f, Pxx = welch(self.x, fs=self.fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, detrend='constant', return_onesided=True)

        if max_hz is None:
            max_hz = self.fs / 2.0
        # band-limit
        mask = (f >= min_hz) & (f <= max_hz)
        f = f[mask]
        Pxx = Pxx[mask]

        if Pxx.size == 0:
            return [] if as_text else []

        # convert to dB for readability and compute percent of total band power
        Pxx_db = 10.0 * np.log10(Pxx + 1e-20)
        total_power = np.sum(Pxx) + 1e-20

        # find peaks and select top-k by linear power
        peaks, _ = find_peaks(Pxx)
        if peaks.size == 0:
            return [] if as_text else []

        peaks = np.asarray(peaks)
        order = np.argsort(Pxx[peaks])[::-1][:k]
        selected = peaks[order]

        results = []
        for rank, idx in enumerate(selected, start=1):
            freq = float(f[idx])
            power = float(Pxx[idx])
            power_db = float(Pxx_db[idx])
            pct = 100.0 * power / total_power

            # nicer frequency formatting
            if freq >= 1000.0:
                freq_str = f"{freq/1000.0:.2f} kHz"
            else:
                freq_str = f"{freq:.1f} Hz"

            label = f"{rank}) {freq_str} — {power_db:.1f} dB ({pct:.2f}% of band power)"
            entry = {
                'rank': rank,
                'frequency_hz': freq,
                'power_linear': power,
                'power_db': power_db,
                'percent_band_power': pct,
                'label': label
            }
            results.append(entry)

        if show:
            for r in results:
                print(r['label'])

        return [r['label'] for r in results] if as_text else results
