import time
import numpy as np
import logging as log
import yaml
import matplotlib.pyplot as plt
from itertools import islice, cycle
from matplotlib.colors import LogNorm
from scipy.special import erf
import click
import os
import datetime
from multiprocessing import Pool
from functools import partial
import traceback 
import shutil

import make_fake

from sps_common.interfaces import PowerSpectra, DedispersedTimeSeries
from sps_common.interfaces.utilities import harmonic_sum, powersum_at_sigma, sigma_sum_powers
from sps_common.constants import FREQ_TOP, FREQ_BOTTOM, TSAMP, DM_CONSTANT

import sps_pipeline
from sps_pipeline.processing import find_active_pointings

from ps_processes.processes.ps import PowerSpectraCreation
from ps_processes.processes.ps_stack import PowerSpectraStack
from ps_processes.processes.ps_inject import Injection

from scheduler.workflow import schedule_workflow_job, remove_finished_service

from sps_databases import db_api, db_utils

import beamformer.skybeam as bs
from beamformer.utilities.common import find_closest_pointing, get_data_list
from beamformer.strategist.strategist import PointingStrategist

from sps_dedispersion.dedisperse import dedisperse
from dmt.libdmt import FDMT

def get_max_dm(ra, dec):
    db_mode = 'database'
    db = db_utils.connect(host='sps-archiver1', name='test')
    ap = find_closest_pointing(ra, dec, mode=db_mode)
    return ap.maxdm

def dm_distribution(x, mu, sig, l):
    gauss = l*np.exp(l*(2*mu + l*sig**2 - 2*x)/2)/2
    tail = 1 - erf((mu + l*sig**2 - x) / np.sqrt(2) / sig) #complimentary error function

    return gauss*tail / np.sum(gauss*tail)

def load_pspec(file_path):

    pspec = PowerSpectra.read(file_path)
    pspec_median = np.median(pspec.power_spectra)
    pspec_mean = np.mean(pspec.power_spectra)
    f_nyquist = pspec.freq_labels[-1]
    df = (f_nyquist - pspec.freq_labels[0]) / len(pspec.freq_labels)
    print(f'Synthetic power spectrum loaded with median = {pspec_median} and mean = {pspec_mean}.') 
    
    return pspec

def get_injections(N, maxdm):
    
    f_nyquist = 508
    TPA_profiles = np.load(f'{transmissivity_repo_path}/profiles/smoothed_baselined_TPA_pulses.npy')
    prof_idx = np.random.choice(range(len(TPA_profiles)), N)

    f_dist = np.loadtxt(f'{transmissivity_repo_path}/scripts/atnf_freqs.txt', usecols = [1])
    f_log = np.logspace(-3, 2.7, int((4/6)*len(f_dist)))
    f_choices = np.concatenate([f_dist, f_log]) 
    f_choices = f_choices[f_choices < f_nyquist]
    f = np.random.choice(f_choices, size = N)
    
    dm_spread = np.linspace(0, maxdm, 10000)
    dm_weights = dm_distribution(dm_spread, 24, 24, 0.02)
    #24 is chosen as the maximum DM value at b = 90 deg from NE2001
    dm_dist = np.random.choice(dm_spread, size = int(0.6*N), p = dm_weights)
    dm_linear = np.linspace(0, maxdm, int(0.4*N))
    dm = np.concatenate([dm_dist, dm_linear])
    
    S_choices = np.logspace(-2, 1, 10000)
    S = np.random.choice(S_choices, N)


    return prof_idx, f, dm, S

def get_injections(N, maxdm, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    f_nyquist = 508
    TPA_profiles = np.load(f'{transmissivity_repo_path}/profiles/smoothed_baselined_TPA_pulses.npy')
    prof_idx = rng.choice(range(len(TPA_profiles)), N)
    
    f_dist = np.loadtxt('atnf_freqs.txt', usecols=[1])
    f_log = np.logspace(-3, 2.7, int((4/6)*len(f_dist)))
    f_choices = np.concatenate([f_dist, f_log])
    f_choices = f_choices[f_choices < f_nyquist]
    f = rng.choice(f_choices, size=N)
    
    dm_spread = np.linspace(0, maxdm, 10000)
    dm_weights = dm_distribution(dm_spread, 24, 24, 0.02)
    #24 is chosen as the maximum DM value at b = 90 deg from NE2001
    dm_dist = rng.choice(dm_spread, size=int(0.6*N), p=dm_weights)
    dm_linear = np.linspace(0, maxdm, int(0.4*N))
    dm = np.concatenate([dm_dist, dm_linear])
    
    S_choices = np.logspace(-2, 1, 10000)
    S = rng.choice(S_choices, N)
    
    return prof_idx, f, dm, S

def call_and_retrieve(pointing, N_days, ii, prof_idx, f, dm, S):
    ra = np.round(pointing[0], 2)
    dec = np.round(pointing[1], 2)
    sub_pointing = int(pointing[2])
    period_string = "_".join([f"{f[i]:.2f}" for i in ii])
    instant_time = time.time()
    temp_path = f'{transmissivity_repo_path}/scripts/inj_{instant_time}_{period_string}_{ra}_{dec}.yaml'
    make_fake.make_yaml(len(ii), temp_path, 'tpa', None, prof_idx[ii], f[ii], dm[ii], S[ii])
    print('Made fakes.')
    cand_path = f'{cand_directory_path}/{N_days}/{ra:.2f}_{dec:.2f}_{sub_pointing}_{injection_path.split('/')[-1]}_{str(injection_idx).replace(' ', '')}_{period_string}_injection_candidates.npz'
    print(f'cand path: {cand_path}')
    print('Attempting to inject.')
    if dec < 0:
        dec_string = f'" -{str(np.abs(dec))}"'
    else:
        dec_string = str(dec)
    print('Running call.') 

    #---MAY NEED TO ADD DB CONFIG TO THIS CALL---#
    #the candidate should go to a subdirectory /Ndays/
    os.system(f'run-stack-search-pipeline {str(ra)} {dec_string} search-monthly --cand-path ./{N_days}/ \
            --injection-path {temp_path} --only-injections')
    try:
        cands = np.load(cand_path, allow_pickle = True)
        print('Loaded candidate file.')
    except:
        print('Could not load candidate file.')

    output = np.zeros((len(ii), 14))
    output[:, 0:3] = np.array([ra, dec, N_days])[np.newaxis, :]

    cand_freqs = []
    cand_idx = []
    for i in range(cands['cand_count'].item()):
        cand = cands[f'candidate_{i}'].item()
        cand_freqs.append(cand['freq'])
        if cand['injection']:
            cand_idx.append(cand['injection_dict']['injection_index'])
        else:
            cand_idx.append(-1)

    cand_freqs = np.asarray(cand_freqs)
    cand_idx = np.asarray(cand_idx)
    if len(cand_freqs) > 0:
        print(f'Candidate frequencies are {[np.round(i, 3) for i in cand_freqs]} Hz.')
    
    for i in range(len(ii)):
        inj = cands['injection_dicts'][i]
        #note that fwhm is totally borked in the output file
        #this is okay, we can recover it from the TPA_idx
        output[i, 3:10] = (prof_idx[ii[i]], inj['frequency'], inj['DM'], inj['flux'], 0.1, inj['predicted_sigma'], inj['predicted_nharm'])
        
        if len(cand_freqs) > 0:
            same_injection = np.where(cand_idx == i)[0]
            closest_match = np.argmin(np.abs(cand_freqs - f[ii[i]])) 
            #if np.abs(cand_freqs[closest_match] - f[ii[i]]) <= 0.01: 
            if closest_match in same_injection:
                cand = cands[f'candidate_{closest_match}'].item()
                output[i, 10:14] = (cand['sigma'], cand['features'].item()[3], cand['freq'], cand['dm'])
                print(f'Matched candidate with frequency {cand_freqs[closest_match]:.3f} to injection with frequency {f[ii[i]]:.3f}.')
            else:
                output[i, 10:14] = -1 * np.ones(4)
                print(f'There may be something concerning going on with f = {f[ii[i]]:.3f}, DM = {dm[ii[i]]:.2f}, and S = {S[ii[i]]:.2f}.')

        else:
            output[i, 10:14] = -1 * np.ones(4)
            print(f'Did not find a match for injection with f = {f[ii[i]]:.3f}, DM = {dm[ii[i]]:.2f}, and S = {S[ii[i]]:.2f}.')
    
    #output file has columns:
    #RA, Dec, Date, TPA_idx, f, DM, flux, fwhm, predicted_sigma, predicted_nharm, output_sigma, output_nharm, output_f, output_dm
    
    shutil.move(cand_path, f'{move_cands_to}/{ra}_{dec}_{N_days}_cands_{ii}.npz')
    os.remove(temp_path)

    return output
        
def inject(N_days, N, pointing, seed):
    ra = pointing[0]
    dec = pointing[1]
    #mode = 'database'
    #ap = find_closest_pointing(ra, dec, mode=mode)   
    ra = np.round(ra, 2)
    dec = np.round(dec, 2)
    print(f'Injecting {N} pulsars into ({ra}, {dec}) on {date}.')
    #kernels = np.load(os.path.dirname(__file__) + "/kernels.npz")    
    maxdm = get_max_dm(ra, dec)
    #ensure randomization between workers
    rng = np.random.default_rng(seed)
    prof_idx, f, dm, S = get_injections(N, maxdm, rng = rng)

    main_injection_idx = np.where(S < 1.)[0]
    separate_injection_idx = np.where(S >= 1.)[0]
    print(f'main idx = {main_injection_idx}')
    print(f'sep idx = {separate_injection_idx}')

    try:
        #all run in block in same call        
        call_and_retrieve(pointing, date, main_injection_idx, prof_idx, f, dm, S)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc() 

    for ii in separate_injection_idx:
        try:    
            #run individually to avoid clustering issues
            call_and_retrieve(pointing, date, np.array([ii]), prof_idx, f, dm, S)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc() 

pointings = np.load(f'{transmissivity_repo_path}/scripts/20251107_pointings.npy') #assume pointings in stack are close to 11/07 pointings
print('Loaded pointings.')

#---THINGS THAT NEED TO BE TWEAKED FOR RUNNING ON NARVAL---#
#there's some stuff in call_and_retrieve() that may also need to be tweaked
#I indicated those spots with #---# 
N_days = np.array([3, 6, 9, 12, 15, 18, 21])

cand_directory_path = 'wherever/injections/cands/end/up/on/narval'
move_cands_to = 'path/for/storing/cands'
transmissivity_repo_path = '/path/to/Transmissivity'
#---#

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
#i have no clue whether this will run in order of pointings...
num_jobs = len(pointings)
n_Ndays = len(N_days)

print(f'The number of jobs is {num_jobs * n_Ndays}.')
pointing_idx = task_id % n_pointings
nday_idx     = task_id // n_pointings

pointing = pointings[pointing_idx]
N_day    = N_days[nday_idx]
seed = np.random.SeedSequence(42).spawn(n_pointings * n_ndays)[task_id]
result = inject(np.array([N_day]), 10, pointing, seed)
np.save(f"{transmissivity_repo_path}/results/stack/result_{task_id}.npy", result)
