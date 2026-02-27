ecg_hpf = butter_filter(ecg_noisy, cutoff=0.5, fs=sampling_rate, btype='high')

plt.figure(figsize=(10,5))
plt.plot(time, ecg_noisy, label="Noisy ECG", alpha=0.4)
plt.plot(time, ecg_hpf, label="High-Pass Filtered", linewidth=2)
plt.title("High-Pass Filtered ECG (0.5 Hz)")
plt.xlim(0,5)
plt.legend()
plt.grid()
plt.show()

fft_hpf = np.fft.rfft(ecg_hpf)

plt.figure(figsize=(10,4))
plt.plot(fft_freq, np.abs(fft_hpf))
plt.title("Frequency Spectrum after High-Pass Filtering")
plt.xlim(0,100)
plt.grid()
plt.show()
