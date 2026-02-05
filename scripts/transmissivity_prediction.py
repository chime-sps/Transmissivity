import numpy as np
from ps_processes.processes.ps_stack import PowerSpectraStack
from ps_processes.processes.ps_inject import Injection
from sps_common.interfaces import PowerSpectra, DedispersedTimeSeries
from ps_processes.processes.ps import PowerSpectraCreation
import logging as log
from sps_common.interfaces.utilities import (
    harmonic_sum,
    powersum_at_sigma,
    sigma_sum_powers,
)
import yaml
import matplotlib.pyplot as plt
from sps_common.constants import FREQ_TOP, FREQ_BOTTOM, TSAMP, DM_CONSTANT
from sps_databases import db_api, db_utils
import beamformer.skybeam as bs
from sps_dedispersion.dedisperse import dedisperse
from dmt.libdmt import FDMT
from matplotlib.colors import LogNorm
from scipy.special import erf

def dm_distribution(x, mu, sig, l):
    gauss = l*np.exp(l*(2*mu + l*sig**2 - 2*x)/2)/2
    tail = 1 - erf((mu + l*sig**2 - x) / np.sqrt(2) / sig) #complimentary error function

    return gauss*tail / np.sum(gauss*tail)

pspec = PowerSpectra.read('blank_pspec.hdf5')
pspec_median = np.median(pspec.power_spectra)
pspec_mean = np.mean(pspec.power_spectra)
f_nyquist = pspec.freq_labels[-1]
df = (f_nyquist - pspec.freq_labels[0]) / len(pspec.freq_labels)
print(f'Synthetic power spectrum loaded with median = {pspec_median} and mean = {pspec_mean}.') 
kernels = np.load('/home/squillace/champss_software/champss/ps-processes/ps_processes/processes/kernels.npy')
kernel_scaling = np.load('/home/squillace/champss_software/champss/ps-processes/ps_processes/processes/kernels.meta.npy')

profiles = np.load('smoothed_baselined_TPA_pulses.npy')
N_batch = 300
N = len(profiles)
output = np.zeros((N*N_batch, 7))

with open('synthetic_map_with_tsky.txt', 'a') as file:
    for batch in range(N_batch):
        
        f_dist = np.loadtxt('atnf_freqs.txt', usecols = [1])
        f_log = np.logspace(-3, 2.7, int((4/6)*len(f_dist)))
        f_choices = np.concatenate([f_dist, f_log]) 
        f = np.random.choice(f_choices, size = N)
        f[f > f_nyquist] = f_nyquist

        dm_spread = np.linspace(0, 1000, 10000)
        dm_weights = dm_distribution(dm_spread, 24, 24, 0.02)
        #24 is chosen as the maximum DM value at b = 90 deg from NE2001
        dm_dist = np.random.choice(dm_spread, size = int(0.6*N), p = dm_weights)
        dm_linear = np.linspace(0, 1000, int(0.4*N))
        dm = np.concatenate([dm_dist, dm_linear])
        dm[dm > pspec.dms[-1]] = pspec.dms[-1]

        S_choices = np.logspace(-2, 1, 10000)
        S = np.random.choice(S_choices, N)
        #TPA_idx, f, DM, flux, fwhm, predicted_sigma, predicted_nharm

        padded_length = 2 * pspec.power_spectra.shape[1]
        num_harm = 32
        full_harm_bins = np.vstack(
                        (
                            np.arange(0, padded_length // 2),
                            harmonic_sum(num_harm, np.zeros(padded_length // 2))[1],
                        )
                    ).astype(np.int32)
        for i in range(N):

            print(f'f = {f[i]} Hz, DM = {dm[i]} pc / cm^3, S = {S[i]} mJy.')
            injection = Injection(pspec, full_harm_bins, dm[i], f[i], profiles[i],
                    flux = S[i])

            fwhm = injection.W * f[i]
            print(f'ACF routine fit FWHM = {np.round(100 * fwhm)}%.')
            if fwhm > 0.5:
                print('ACF too wide! Skipping this injection.')
                continue

            scaled_prof_fft, phases = injection.flux_to_power()
            smeared_prof_fft = injection.smear_fft(scaled_prof_fft)
            windowed_prof_fft = injection.time_windowing(smeared_prof_fft)

            n_harm = int(np.floor(f_nyquist / f[i]))
            if 32 < n_harm:
                n_harm = 32

            if len(scaled_prof_fft) > n_harm:
                windowed_prof_fft = windowed_prof_fft[:n_harm]
            used_nharm = len(windowed_prof_fft)

            dispersed_prof_fft, dm_indices = injection.disperse(
                windowed_prof_fft, kernels, kernel_scaling
            )
            harms = []
            for k in range(len(dispersed_prof_fft)):
                bins, harm = injection.scalloping(dispersed_prof_fft[k], df)
                harms.append(harm)

            harms = np.asarray(harms)
            normaliser = injection.get_rednoise_normalisation(
                        bins,
                        dm_indices,
                    )
            harms *= normaliser
            
            try:
                (
                    harms,
                    predicted_nharm,
                    predicted_sigma,
                    rescale_factor,
                ) = injection.predict_sigma(harms, bins, dm_indices, used_nharm, True)
            except:
                continue
            info = f'{i} {f[i]} {dm[i]} {S[i]} {fwhm} {predicted_sigma} {predicted_nharm}\n'
            file.write(info)
