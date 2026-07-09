import time
import numpy as np
import logging as log
import yaml
import json
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

    maxdm = maxdm_dict[f'{ra} {dec}']
    print(f'Found maximum DM: {maxdm}.')
    return maxdm

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

def get_injections(N, maxdm, rng=None):
    
    if rng is None:
        rng = np.random.default_rng()

    f_nyquist = 508
    TPA_profiles = np.load(f'{transmissivity_repo_path}/profiles/smoothed_baselined_TPA_pulses.npy')
    prof_idx = rng.choice(range(len(TPA_profiles)), N)
    
    f_dist = np.loadtxt(f'{transmissivity_repo_path}/scripts/atnf_freqs.txt', usecols=[1])
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
    
    S_choices = np.logspace(-3, 1, 10000)
    S = rng.choice(S_choices, N)
    
    return prof_idx, f, dm, S

def call_no_retrieve(pspec_file, ii, prof_idx, f, dm, S):
    ra = float(pspec_file.split('_')[0])
    dec = float(pspec_file.split('_')[1])
    period_string = "_".join([f"{f[i]:.2f}" for i in ii])
    instant_time = time.time()
    temp_path = f'{transmissivity_repo_path}/scripts/inj_{instant_time}_{period_string}_{ra}_{dec}.yaml'
    make_fake.make_yaml(len(ii), temp_path, 'tpa', None, prof_idx[ii], f[ii], dm[ii], S[ii])
    print('Made fakes.')

    print('Attempting to inject.')
    if dec < 0:
        dec_string = f'" -{str(np.abs(dec))}"'
    else:
        dec_string = str(dec)
    print('Running call.') 

    #---NARVAL TWEAKS---#
    power_spectra_path = f'/project/ctb-vkaspi/champss/stack_variable_depth/saved_stacks/{pspec_file}'
    #power_spectra_path = f'/project/ctb-vkaspi/champss/stack_202511/{ra}_{dec}_power_spectra_stack.hdf5'
    #---#

    os.system(f'run-stack-search-pipeline {str(ra)} {dec_string} search-monthly --cand-path {cand_directory_path} \
            --file {power_spectra_path} --injection-path {temp_path} --only-injections')
    #output file has columns:
    #RA, Dec, Date, TPA_idx, f, DM, flux, fwhm, predicted_sigma, predicted_nharm, output_sigma, output_nharm, output_f, output_dm
    
    os.remove(temp_path)

    return
        
def inject(N, pointing_idx, seed):
    rng = np.random.default_rng(seed)
    for idx in range(pointing_idx, pointing_idx + 50):
        pspec_file = pspec_files[idx]
        ra = float(pspec_file.split('_')[0])
        dec = float(pspec_file.split('_')[1])
        #mode = 'database'
        #ap = find_closest_pointing(ra, dec, mode=mode)   
        #ra = np.round(ra, 2)
        #dec = np.round(dec, 2)
        print(f'Injecting {N} pulsars into ({ra}, {dec}).')
        maxdm = get_max_dm(ra, dec)
        #ensure randomization between workers
        prof_idx, f, dm, S = get_injections(N, maxdm, rng = rng)

        main_injection_idx = np.where(S < 1.)[0]
        separate_injection_idx = np.where(S >= 1.)[0]
        print(f'main idx = {main_injection_idx}')
        print(f'sep idx = {separate_injection_idx}')

        if len(main_injection_idx) > 0:
            try:
                #all run in block in same call        
                call_no_retrieve(pspec_file, main_injection_idx, prof_idx, f, dm, S)
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc() 

        for ii in separate_injection_idx:
            try:    
                #run individually to avoid clustering issues
                call_no_retrieve(pspec_file, np.array([ii]), prof_idx, f, dm, S)
            except Exception as e:
                print(f"An error occurred: {e}")
                traceback.print_exc() 


#---THINGS THAT NEED TO BE TWEAKED FOR RUNNING ON NARVAL---#
#there's some stuff in call_and_retrieve() that may also need to be tweaked
#I indicated those spots with #---#
cand_directory_path = '/project/ctb-vkaspi/champss/injections'
transmissivity_repo_path = '/project/ctb-vkaspi/champss/injections/Transmissivity'
#---#
with open('stack_maxdm.json', 'r') as f:
    maxdm_dict = json.load(f)
#pointings = np.load(f'{transmissivity_repo_path}/scripts/stack_pointings.npy')
#pointings = pointings
#print('Loaded pointings.')
with open(f'{transmissivity_repo_path}/scripts/stack_var_files.txt', 'r') as f:
    pspec_files = f.readlines()
batch_indices = np.arange(len(pspec_files))[::50]  #start idx of each batch
seeds = np.random.SeedSequence(42).spawn(len(batch_indices))

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])  #range over len(batch_indices)
seed = seeds[task_id]
inject(10, batch_indices[task_id], seed)
#np.save(f"{transmissivity_repo_path}/results/stack/result_{task_id}.npy", result)
