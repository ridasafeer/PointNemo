from class_lib.plotter import WavAnalysis

wa = WavAnalysis("audio_data/microwave.wav")
wa.plot_power_spectrum()
print(wa.top_frequencies(k=5, min_hz=20))