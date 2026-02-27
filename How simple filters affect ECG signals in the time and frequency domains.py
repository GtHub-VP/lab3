def butter_filter(data, cutoff, fs, btype, order=4):
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype=btype)
return filtfilt(b, a, data)

ecg_lpf = butter_filter(ecg_noisy, cutoff=40, fs=sampling_rate, btype='low')

plt.figure(figsize=(10,5))
plt.plot(time, ecg_noisy, label="Noisy ECG", alpha=0.5)
plt.plot(time, ecg_lpf, label="Low-Pass Filtered", linewidth=2)
plt.title("Low-Pass Filtered ECG (40 Hz)")
plt.xlim(0,5)
plt.legend()
plt.grid()
plt.show()

# FFT
fft_lpf = np.fft.rfft(ecg_lpf)

plt.figure(figsize=(10,4))
plt.plot(fft_freq, np.abs(fft_lpf))
plt.title("Frequency Spectrum after Low-Pass Filtering")
plt.xlim(0,100)
plt.grid()
plt.show()
