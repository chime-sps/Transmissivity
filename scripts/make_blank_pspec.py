import numpy as np
from ps_processes.processes.ps_stack import PowerSpectraStack
from sps_common.interfaces import PowerSpectra, DedispersedTimeSeries
from ps_processes.processes.ps import PowerSpectraCreation
import logging as log
from sps_common.constants import FREQ_TOP, FREQ_BOTTOM, TSAMP, DM_CONSTANT
from sps_databases import db_api, db_utils
import beamformer.skybeam as bs
from sps_dedispersion.dedisperse import dedisperse
from dmt.libdmt import FDMT
from matplotlib.colors import LogNorm

chunk_size = 2400 * 23
maxdm = 10**3.3
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
pspec.write('blank_pspec_16k.hdf5')
print('Blank power spectra written to blank_pspec_16k.hdf5.')
