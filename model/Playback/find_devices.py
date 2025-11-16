import sounddevice as sd

print("Output devices only:\n")
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_output_channels'] > 0:
        print(f"Device {i}: {device['name']}")
        print(f"  Output Channels: {device['max_output_channels']}")
        print(f"  Sample Rate: {device.get('default_samplerate', 'N/A')} Hz\n")