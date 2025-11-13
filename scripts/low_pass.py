import numpy as np
from scipy.fft import fft, irfft
import matplotlib.pyplot as plt

profiles = np.load('TPA_profiles.npy')

def smooth(prof, cutoff):

    prof_fft = fft(prof)
    freq = np.arange(len(prof))
    smoothed_fft = np.zeros(len(prof_fft), dtype = "complex_")
    keep = prof_fft[freq < cutoff]
    smoothed_fft[:len(keep)] = keep

    return irfft(smoothed_fft)

prof = profiles[0]
prof /= max(prof)
smoothed_prof = smooth(prof, 50)
print(max(smoothed_prof))
smoothed_prof /= max(smoothed_prof)
print(max(smoothed_prof))
phi = np.linspace(0, 1, 1024)
plt.plot(phi, prof)
plt.plot(phi, smoothed_prof)
plt.show()
