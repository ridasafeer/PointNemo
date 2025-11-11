import numpy as np
_HAVE_SCIPY = True
try:
    from scipy.io import wavfile
except Exception:
    _HAVE_SCIPY = False
    import wave

def read_wav_mono(path):
    if _HAVE_SCIPY:
        fs, x = wavfile.read(path)
        if x.dtype == np.int16:  x = x.astype(np.float32) / 32768.0
        elif x.dtype == np.int32: x = x.astype(np.float32) / 2147483648.0
        elif x.dtype == np.uint8: x = (x.astype(np.float32)-128)/128.0
        else:                     x = x.astype(np.float32)
        if x.ndim == 2: x = x.mean(axis=1)
        return fs, x
    # minimal 16-bit fallback
    with wave.open(path, 'rb') as wf:
        nch, sw, fs = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
        nframes, raw = wf.getnframes(), wf.readframes(wf.getnframes())
    if sw != 2: raise RuntimeError("Only 16-bit PCM supported without SciPy.")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch == 2: x = x.reshape(-1,2).mean(axis=1)
    return fs, x

def write_wav(path, fs, x, float32=False):
    y = np.asarray(np.clip(x, -1.0, 1.0), dtype=np.float32)
    if _HAVE_SCIPY and float32:
        wavfile.write(path, fs, y)                        # 32-bit float WAV
    elif _HAVE_SCIPY:
        wavfile.write(path, fs, (y*32767.0).astype(np.int16))
    else:
        import wave
        data = (y*32767.0).astype(np.int16)
        with wave.open(path,'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
            wf.writeframes(data.tobytes())

def resample_to(x, fs_in, fs_out):
    if fs_in == fs_out: return x.astype(np.float32)
    import numpy as np
    dur = len(x)/fs_in
    t_old = np.linspace(0, dur, num=len(x), endpoint=False)
    t_new = np.linspace(0, dur, num=int(round(dur*fs_out)), endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)
