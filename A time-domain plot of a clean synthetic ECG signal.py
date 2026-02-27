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
