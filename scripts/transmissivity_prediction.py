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

profiles = np.load('smooth_TPA_pulses.npy')
N = len(profiles)

f_choices = np.loadtxt('atnf_freqs.txt', usecols = [1])
f = np.random.choice(f_choices, size = N)

dm_choices = np.linspace(0, 10**3.3, 10000)
dm_weights = dm_distribution(dm_choices, 24, 24, 0.02)
#24 is chosen as the maximum DM value at b = 90 deg from NE2001
#0.02 is chosen as the scale where p(w > 0) > 99.7%
dm = np.random.choice(dm_choices, size = N, p = dm_weights)

S = np.random.uniform(0.01, 1., N)

chunk_size = 2400 * 23
maxdm = 1000
nchan = 2*1024
# use log-spacing from slow to near Nyquist, but at integer FFT bins
db = db_utils.connect(host='sps-archiver1', name='test')
num_dms_fac = 1
num_threads = 16 # parallelize, if desired
num_dms = (
    int(
        DM_CONSTANT
        * maxdm
        * (1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2)
        / TSAMP
        // num_dms_fac
    )
    + 1
) * num_dms_fac
nsamps = chunk_size + num_dms
t = np.arange(nsamps) * TSAMP

# define fdmt
fdmt = FDMT(FREQ_BOTTOM, FREQ_TOP, nchan, nsamps, TSAMP, dt_max=num_dms, dt_min=0, dt_step=1)
# multithread, if desired
if num_threads > 1:
    fdmt.set_num_threads(num_threads)
# run fdmt
dm_step = 1

data = 1.0 * np.random.standard_normal((nchan, nsamps)).astype(np.float32)
sb = bs.SkyBeam(spectra=data, ra=50.814417767387674, dec=39.90059501729749, nchan=nchan, ntime=nsamps, maxdm=maxdm, beam_row=103, utc_start=1696587658.0, obs_id='67f574ba78c208d7fbbf586f', pointing_id=None, nbits=32)
dts = dedisperse(fdmt, sb, chunk_size, num_dms, dm_step)

pspec = PowerSpectraCreation(
        clean_rfi = False,
        run_static_filter = False,
        run_dynamic_filter = False,
        update_db = False,
).transform(dts)
pspec_median = np.median(pspec.power_spectra)
print(f'Synthetic power spectrum created with median = {pspec_median}.') 
kernels = np.load('/home/squillace/champss_software/champss/ps-processes/ps_processes/processes/kernels.npy')
kernel_scaling = np.load('/home/squillace/champss_software/champss/ps-processes/ps_processes/processes/kernels.meta.npy')

padded_length = 2 * pspec.power_spectra.shape[1]
num_harm = 32
full_harm_bins = np.vstack(
                (
                    np.arange(0, padded_length // 2),
                    harmonic_sum(num_harm, np.zeros(padded_length // 2))[1],
                )
            ).astype(np.int32)

#TPA_idx, f, DM, flux, fwhm, predicted_sigma, predicted_nharm
output = np.zeros((N, 7))

for i in range(N):

    print(f'f = {f[i]} Hz, DM = {dm[i]} pc / cm^3, S = {S[i]} mJy.')
    injection = Injection(pspec, full_harm_bins, dm[i], f[i], profiles[i],
            flux = S[i])

    fwhm = injection.W
    print(f'ACF routine fit FWHM = {100 * fwhm * f[i]}%.')
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
    
    (
        harms,
        predicted_nharm,
        predicted_sigma,
        rescale_factor,
    ) = injection.predict_sigma(harms, bins, dm_indices, used_nharm, True)

    output[i, :] = np.asarray([i, f, dm, S, fwhm, predicted_sigma, predicted_nharm])

np.save('synthetic_map1.npy', output)
