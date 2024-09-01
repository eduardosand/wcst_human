import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from intracranial_ephys_utils.preprocess import broadband_seeg_processing

# Parameters
duration = 3.0  # duration in seconds
fs_32k = 32000  # sampling rate for the first stimulus
fs_8k = 8000  # sampling rate for the second stimulus
step_time = 1.5  # time where the step occurs in seconds
lowfreq = 1  # low frequency for bandpass filter (in Hz)
highfreq = 100  # high frequency for bandpass filter (in Hz)

# Frequencies for the sine waves
freq1 = 5  # in Hz
freq2 = 30 # in Hz
# Generate stimulus sampled at 32KHz
t_32k = np.linspace(0, duration, int(duration * fs_32k), endpoint=False)
# stimulus_32k = np.where(t_32k >= step_time, 1, 0)
stimulus_32k = np.where(t_32k >= step_time, np.sin(2 * np.pi * freq1 * t_32k), 0) + np.sin(2 * np.pi * freq2 * t_32k)

# Generate stimulus sampled at 8KHz
t_8k = np.linspace(0, duration, int(duration * fs_8k), endpoint=False)
# stimulus_8k = np.where(t_8k >= step_time, 1, 0)
stimulus_8k = np.where(t_8k >= step_time, np.sin(2 * np.pi * freq1 * t_8k), 0) + np.sin(2 * np.pi * freq2 * t_8k)

# Generate stimulus sampled at 4KHz
fs_4k = 4000  # sampling rate for the second stimulus
t_4k = np.linspace(0, duration, int(duration * fs_4k), endpoint=False)
# stimulus_8k = np.where(t_8k >= step_time, 1, 0)
stimulus_4k = np.where(t_4k >= step_time, np.sin(2 * np.pi * freq1 * t_4k), 0) + np.sin(2 * np.pi * freq2 * t_4k)


# Process both signals
low_pass = 1000
processed_32k, fs_processed_32k = broadband_seeg_processing(stimulus_32k, fs_32k, 0.1, low_pass)
processed_8k, fs_processed_8k = broadband_seeg_processing(stimulus_8k, fs_8k, 0.1, low_pass)
processed_4k, fs_processed_4k = broadband_seeg_processing(stimulus_4k, fs_4k, 0.1, low_pass)

# Time vectors for the downsampled signals
t_processed_32k = np.linspace(0, duration, len(processed_32k), endpoint=False)
t_processed_8k = np.linspace(0, duration, len(processed_8k), endpoint=False)
t_processed_4k = np.linspace(0, duration, len(processed_4k), endpoint=False)

# Plotting the original and processed signals
plt.figure(figsize=(12, 8))

# Original signals
plt.subplot(2, 3, 1)
plt.plot(t_32k, stimulus_32k, label='Original 32KHz')
plt.title('Original 32KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(t_8k, stimulus_8k, label='Original 8KHz')
plt.title('Original 8KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(t_4k, stimulus_4k, label='Original 8KHz')
plt.title('Original 8KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Processed signals
plt.subplot(2, 3, 4)
plt.plot(t_processed_32k, processed_32k, label='Processed 32KHz')
plt.title('Processed 32KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(t_processed_8k, processed_8k, label='Processed 8KHz')
plt.title('Processed 8KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(t_processed_4k, processed_4k, label='Processed 8KHz')
plt.title('Processed 4KHz Stimulus')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()