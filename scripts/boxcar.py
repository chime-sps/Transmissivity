import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

profiles = np.load('TPA_profiles.npy')

def smooth(prof, width):

    prof_fft = fft(prof)
    boxcar = np.zeros(len(prof))
    boxcar[:width] = 1
    boxcar_fft = fft(boxcar)
    conv_fft = prof_fft * boxcar_fft
    conv = ifft(conv_fft)

    return conv

def baseline(smoothed_prof, off_center = 300):

    mean1 = np.mean(smoothed_prof[:off_center])
    mean2 = np.mean(smoothed_prof[-off_center:])
    mean = (mean1 + mean2) / 2
    smoothed_prof -= mean

    return smoothed_prof

def get_rms(smoothed_prof, off_center = 300):

    mean_squared1 = np.mean(np.abs(smoothed_prof[:off_center])**2)
    mean_squared2 = np.mean(np.abs(smoothed_prof[-off_center:])**2)
    mean_squared = (mean_squared1 + mean_squared2) / 2

    return np.sqrt(mean_squared)

prof = profiles[0]
prof /= max(prof)
width = 25
smoothed_prof = smooth(prof, width)
smoothed_prof /= max(smoothed_prof)
smoothed_prof = baseline(smoothed_prof)
off_center = 300
rms = get_rms(smoothed_prof)
print(f'rms = {rms}')

phi = np.linspace(0, 1, 1024)
plt.plot(phi, prof)
plt.plot(phi, smoothed_prof)
plt.vlines([phi[off_center], phi[-off_center]], ymin = 0, ymax = 1, color = 'gray', linestyle = '--', label = 'Baseline region')
plt.title(f"Boxcar with width = {width} bins")
plt.legend()
plt.show()
