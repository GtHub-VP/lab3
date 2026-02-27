noise = 0.3 * np.random.randn(len(ecg))

ecg_noisy = ecg + noise

plt.figure(figsize=(10,4))
plt.plot(time, ecg_noisy)
plt.title("Noisy ECG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, 5)
plt.grid()
plt.show()
