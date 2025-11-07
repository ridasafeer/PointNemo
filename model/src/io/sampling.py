###
#Setting sampling rate for sampling the input signal
#Conversion of analog data to digital data type that will be used by algorithm
#Upsampling/downsampling that might be required
#Scaling the signal if needed

def read_wav_mono(path): # Read a WAV file and convert to mono float32 array
    fs, x = wavfile.read(path) #read .wav file; {fs: sampling rate (samples/sec), x: raw audio samples as an integer NumPy array}

    #depending on the data type of the input audio, scale it to float32 in range [-1.0, 1.0]
    if x.dtype == np.int16:
        y = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        y = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.uint8:
        y = (x.astype(np.float32) - 128) / 128.0
    else:
        y = x.astype(np.float32)

    #if stereo, convert to mono by averaging channels
    if y.ndim == 2:
        y = y.mean(axis=1)

    #return the sampling rate and the mono float32 array
    return fs, y

def write_wav(path, fs, x):
    x = np.clip(x, -1.0, 1.0) # Ensure signal is in [-1.0, 1.0] range

    # Write a float32 array to a WAV file as int16
    # Reverse of normalization: scale float [-1, +1] to integer [-32768, +32767]
    wavfile.write(path, fs, (x * 32767.0).astype(np.int16))