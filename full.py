import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from scipy.signal import butter, filtfilt

sampling_rate = 1000  
duration = 10  

ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate)

time = np.linspace(0, duration, len(ecg))

plt.figure(figsize=(10,4))
plt.plot(time, ecg)
plt.title("Clean Synthetic ECG Signal (1000 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, 10)
plt.grid()
plt.show()

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

# Observations at:1000 Hz: Sharp, detailed QRS complex, 500 Hz: Slight smoothing, 250 Hz: Peaks less sharp, 125 Hz: QRS begins to distort. Features become unclear around 125 Hz, specifically sharp R-peaks.

fft_vals = np.fft.rfft(ecg)
fft_freq = np.fft.rfftfreq(len(ecg), 1/sampling_rate)

plt.figure(figsize=(10,4))
plt.plot(fft_freq, np.abs(fft_vals))
plt.title("Frequency Spectrum of Clean ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100)
plt.grid()
plt.show()

# Majority of ECG energy is below 40 Hz. The ECG primarily occupies low frequencies due to slow changes in heart activity and sharp QRS complexes still remaining under ~40 Hz


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

# The feature most affected by noise is the small features like the P and T waves, QRS peaks become less sharp, and baseline becomes irregular

fft_noisy = np.fft.rfft(ecg_noisy)

plt.figure(figsize=(10,5))
plt.plot(fft_freq, np.abs(fft_vals), label="Clean ECG")
plt.plot(fft_freq, np.abs(fft_noisy), label="Noisy ECG", alpha=0.7)
plt.title("Frequency Spectrum: Clean vs Noisy ECG")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 100)
plt.legend()
plt.grid()
plt.show()

# Noise spreads energy across higher frequencies, the clean ECG spectrum concentrated < 40 Hz, while the noisy ECG shows broadband frequency increases

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

# High-frequency noise was greatly reduced, ECG waveform smoother, and the frequency spectrum shows reduction above 40 Hz

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

# The baseline drift was removed, slow fluctuations were eliminated. However  ECG remains recognizable, with the very low frequencies (those less than 0.5 Hz) suppressed