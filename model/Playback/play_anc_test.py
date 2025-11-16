import os
import sys
import numpy as np
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.src.pn_io.wav import read_wav_mono

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

def list_devices():
    """List all available audio devices."""
    print("\nAvailable audio devices:")
    print("="*60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
        print(f"   Channels: {device['max_output_channels']}")
        print(f"   Default: {device.get('default_samplerate', 'N/A')} Hz\n")
    print("="*60)

def play_on_device(audio, fs, device_id, label):
    """Play audio on specific device."""
    try:
        print(f"  Playing {label} on device {device_id}...")
        sd.play(audio, fs, device=device_id)
        sd.wait()
        print(f"  ✓ {label} finished")
    except Exception as e:
        print(f"  ✗ Error playing {label}: {e}")

def main():
    wavdir = 'model/output_auido'
    
    original_file = os.path.join(wavdir, 'hvac_original.wav')
    anti_file = os.path.join(wavdir, 'hvac_anti_noise.wav')
    
    # Check files exist
    if not os.path.exists(original_file):
        print(f"ERROR: {original_file} not found!")
        sys.exit(1)
    if not os.path.exists(anti_file):
        print(f"ERROR: {anti_file} not found!")
        sys.exit(1)
    
    # Load audio
    print("Loading audio files...")
    fs1, x_orig = read_wav_mono(original_file)
    fs2, x_anti = read_wav_mono(anti_file)
    
    if fs1 != fs2:
        print(f"ERROR: Sample rates don't match! {fs1} vs {fs2}")
        sys.exit(1)
    
    fs = fs1
    
    # Make same length
    max_len = max(len(x_orig), len(x_anti))
    x_orig = np.pad(x_orig, (0, max_len - len(x_orig))).astype(np.float32)
    x_anti = np.pad(x_anti, (0, max_len - len(x_anti))).astype(np.float32)
    
    # Normalize
    max_orig = np.max(np.abs(x_orig))
    max_anti = np.max(np.abs(x_anti))
    if max_orig > 0:
        x_orig = x_orig / (max_orig * 1.1)
    if max_anti > 0:
        x_anti = x_anti / (max_anti * 1.1)
    
    print(f"\n{'='*60}")
    print("SIMULTANEOUS PLAYBACK")
    print(f"{'='*60}")
    print(f"Sample Rate: {fs} Hz")
    print(f"Duration: {len(x_orig)/fs:.2f} seconds\n")
    
    # List devices
    list_devices()
    
    # Get device IDs from user
    print("\nEnter device IDs:")
    speaker_device = input("Laptop speaker device ID (for original): ").strip()
    headphone_device = input("Headphone device ID (for anti-noise): ").strip()
    
    try:
        speaker_device = int(speaker_device)
        headphone_device = int(headphone_device)
    except ValueError:
        print("ERROR: Invalid device IDs!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("Starting simultaneous playback...")
    print(f"Original on speaker (device {speaker_device})")
    print(f"Anti-noise on headphones (device {headphone_device})")
    print(f"Press Ctrl+C to stop.\n")
    
    try:
        # Play both simultaneously using threads
        thread1 = threading.Thread(target=play_on_device, args=(x_orig, fs, speaker_device, "Original"))
        thread2 = threading.Thread(target=play_on_device, args=(x_anti, fs, headphone_device, "Anti-noise"))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        print(f"\n{'='*60}")
        print("Playback completed!")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted.")
        sd.stop()

if __name__ == '__main__':
    main()