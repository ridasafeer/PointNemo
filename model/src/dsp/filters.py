from scipy import signal
import numpy as np

def design_bandpass_fir(fs, f_lo, f_hi, taps): # Design a bandpass FIR filter
    ''' 
    Designs a linear-phase FIR bandpass filter that passes frequencies between f_lo and f_hi Hz and attenuates everything outside that range.
    FIR -> Finite Impulse Response: filter that passes signals within a specific frequency range while attenuating signals outside that range
    '''
    return signal.firwin(taps, [f_lo, f_hi], pass_zero=False, fs=fs, window='hann').astype(np.float32) 


def design_notches_ba(fs, tones, Q=30):
    '''
    Generates IIR notch filters for specific hum tones like 60 Hz, 120 Hz, etc.
    IIR -> Infinite Impulse Response:  output depends on both current and past input values, as well as past output values
    This recursive process gives it an "infinite" impulse response (theoretically never becomes zero, in practice it decays to near zero).
    Suitable for applications like audio equalization and smart sensors where memory is limited
    '''
    """Return a list of (b, a) notch sections. Compatible with older SciPy."""

    sections = [] #list that will store (b, a) IIR filter coefficients for each notch
    for f0 in tones:
        if f0 <= 0 or f0 >= fs/2: #Skip invalid frequencies:
            # Must be > 0 Hz.
            # Must be < Nyquist (fs/2) — cannot notch above Nyquist.
            continue
        try:
            b, a = signal.iirnotch(w0=f0, Q=Q, fs=fs)
        except TypeError:
            w0 = f0/(fs/2.0)
            b, a = signal.iirnotch(w0, Q)
        sections.append((b.astype(np.float64), a.astype(np.float64))) 
    return sections



#TODO COMMENT THIS SHIT TOO
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
