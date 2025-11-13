import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import itertools
from sps_common.constants import TSAMP
from scipy.fft import rfft, irfft, rfftfreq
#import plot_map
from sps_common.constants import DM_CONSTANT, TSAMP, FREQ_TOP, FREQ_BOTTOM
from dmt.libdmt import FDMT
from sps_dedispersion.dedisperse import dedisperse
import beamformer.skybeam as bs
from sps_databases import db_api, db_utils
from ps_processes.utilities.utilities import rednoise_normalise
#import pyfdmt.pyfdmt as pyfdmt

def gaussian(fwhm):
    x = np.linspace(0, 1, 1024)
    sig = fwhm / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * ((x - 0.5) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

@nb.njit
def roll_rows(arr, shifts):
    out = np.empty_like(arr)
    m, n = arr.shape
    
    for i in range(m):
        s = shifts[i] % n      # normalize shift
        
        if s == 0:
            # No shift: just copy row safely
            out[i] = arr[i]
        else:
            # Positive-right roll
            out[i, :s] = arr[i, -s:]
            out[i, s:] = arr[i, :-s]
    
    return out

N_f = 5
N_DM = 5
N_S = 5
N_delta = 5

#N_f = 5
#N_DM = 2
#N_S = 2
#N_delta = 2

chunk_size = 2400 * 23
print(chunk_size * TSAMP)
exit()
print(TSAMP)
print(FREQ_BOTTOM, FREQ_TOP)
print((FREQ_TOP - FREQ_BOTTOM)/2048)
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
Nout = 55202 #c++ fdmt
#Nout = 74964 #pyfdmt
Tout = Nout * TSAMP
t = np.arange(nsamps) * TSAMP
spectral_freqs = np.linspace(FREQ_TOP, FREQ_BOTTOM, nchan)
tau = nsamps * TSAMP
f = np.logspace(-3, 2.7, N_f)
spinbins = ((10**np.linspace(0, 2.7, N_f))*Tout).astype(int)
spinfreqs = spinbins / Tout
DM = np.logspace(0, 3.3, N_DM)
S = np.logspace(-3, 1, N_S)
delta = np.linspace(0.01, 0.5, N_delta)
prof_idx = np.arange(N_delta)
profs = np.zeros((N_f, N_delta, 1024))

# define fdmt
fdmt = FDMT(FREQ_BOTTOM, FREQ_TOP, nchan, nsamps, TSAMP, dt_max=num_dms, dt_min=0, dt_step=1)
# multithread, if desired
if num_threads > 1:
    fdmt.set_num_threads(num_threads)
# run fdmt
dm_step = 2
tstart = 0
tlim = 4096*8 # arbitrary, range for plotting
tlim_plot = tlim*TSAMP

#for i in range(len(f)):
#    t = np.arange(0, 1/f[i], TSAMP)
#    if len(t) > 1024: #highest resolution
#        t = np.linspace(0, 1/f[i], 1024)
#
#    t0 = t[-1] / 2 #can be approximate
#    for j in prof_idx:
#        fwhm = delta[j] / f[i]
#        prof = gaussian(t, t0, fwhm)
#        profs[i, j, :len(prof)] = prof #doesn't need to be centered 
    

def delay_from_DM(DM, freq_emitted):
    """
    Return the delay in seconds caused by dispersion, given
    a Dispersion Measure (DM) in cm-3 pc, and the emitted
    frequency (freq_emitted) of the pulsar in MHz.
    """
    if type(freq_emitted) is type(0.0):
        if freq_emitted > 0.0:
            return DM / (0.000241 * freq_emitted * freq_emitted)
        else:
            return 0.0
    else:
        return np.where(
            freq_emitted > 0.0, DM / (0.000241 * freq_emitted * freq_emitted), 0.0
        )



def get_pows(spinbins, DM, S, delta):
    
    spinfreqs = spinbins / Tout
    data = np.zeros((nchan, nsamps))
    prof = gaussian(delta)
    prof /= max(prof)

    top_delay = delay_from_DM(DM, FREQ_TOP)
    # Add signal with appropriate DM delays
    dm_delays = delay_from_DM(DM, spectral_freqs) - top_delay
    # Note: broadcast the dm_delays across the channels
    argument = 2 * np.pi * (t - dm_delays[:, None]) 
    for i in range(len(spinfreqs)):
        A = S / tau / spinfreqs[i]
        prof *= A
        prof_fft = rfft(prof)
        freq_labels = rfftfreq(1024, d = TSAMP)
        spaced_freqs = rfftfreq(nsamps, d = TSAMP)
        spaced_fft = np.zeros(len(spaced_freqs), dtype = 'complex64')
        j = 0
        while j < len(prof_fft) and j*spinbins[i] < len(spaced_fft):
            spaced_fft[j*spinbins[i]] = prof_fft[j]
            j+= 1
        time_series = irfft(spaced_fft)
        data +=  time_series
        #apply dispersion through shift theorem

        #plt.plot(data[0])
        #plt.show()
        #exit()

        print(f"freq = {spinfreqs[i]:7.3f}, DM = {DM:7.3f}, S = {S:7.3f}, delta = {delta:7.3f}")

    top_delay = delay_from_DM(DM, FREQ_TOP)
    dm_delays = delay_from_DM(DM, spectral_freqs) - top_delay
    #dm_bin_shifts = (dm_delays / TSAMP).astype(int)
    #data = roll_rows(data, dm_bin_shifts)
    #print('Applied DM shifts.')
    noise = 1.0 * np.random.standard_normal((nchan, nsamps)) #rms = 1 mJy before folding
    #data += noise
    data = data.astype(np.float32)
    pp = (A*nchan)**2*Nout**2/4
    
    sb = bs.SkyBeam(spectra=data, ra=50.814417767387674, dec=39.90059501729749, nchan=nchan, ntime=nsamps, maxdm=maxdm, beam_row=103, utc_start=1696587658.0, obs_id='67f574ba78c208d7fbbf586f', pointing_id=None, nbits=32)
    dts = dedisperse(fdmt, sb, chunk_size, num_dms, dm_step)
    Idmt = dts.dedisp_ts
    
    #pyFDMT version:

    #dts = pyfdmt.transform(data, FREQ_TOP, FREQ_BOTTOM, TSAMP, 0, 200)
    #Idmt = dts.data
    DM_idx = np.argmin(np.abs(dts.dms - DM))
    fft = rfft(Idmt[DM_idx])
    power = np.abs(fft)**2
    freq_labels = rfftfreq(chunk_size, d = TSAMP)
    #print(f'DM in: {DM}, DM out: {dts.dms[DM_idx]}')
    #print(f'Nout = {2*len(freq_labels)}')
    #print(f'Tout = {len(freq_labels) * 2 * TSAMP:.2f}s')
    plt.plot(freq_labels, power)
    plt.vlines(freq_labels[spinbins], ymin = 0, ymax = max(power), linestyle = '--', color = 'r')
    plt.xlim(0.5, 2.5)
    plt.show()
    exit()
    ps = np.zeros(len(spinbins))
    for i in range(len(spinbins)):
        ps[i] = np.sum(power[::spinbins[i]])

    return ps


grid = itertools.product((1/spinfreqs)*1000, DM, delta)
for params in grid:
    print(params)
#pows = np.zeros((N_DM * N_S * N_delta, N_f))
#for i in range(N_f):
#    for params in grid:
#        pows[i] = get_pows(spinbins, *params)
exit()
pows = pows.reshape(N_DM, N_S, N_delta, N_f)
np.savez('timestream_pows_test.npz', DM = DM, S = S, delta = delta, f = spinbins / Tout, pows = pows)
