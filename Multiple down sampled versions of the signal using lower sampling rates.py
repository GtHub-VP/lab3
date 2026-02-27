sampling_rates = [1000, 500, 250, 125]

plt.figure(figsize=(12,8))

for i, fs in enumerate(sampling_rates):
    downsample_factor = sampling_rate // fs
    ecg_down = ecg[::downsample_factor]
    time_down = np.linspace(0, duration, len(ecg_down))
    
    plt.subplot(4,1,i+1)
    plt.plot(time_down, ecg_down)
    plt.title(f"ECG at {fs} Hz")
    plt.xlim(0, 10)
    plt.grid()

plt.tight_layout()
plt.show()
