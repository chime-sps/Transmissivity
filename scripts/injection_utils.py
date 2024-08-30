import numpy as np
import yaml
import subprocess

def call(injection_path, ra, dec, ii = None):


    command = ['run-stack-search-pipeline', str(ra), str(dec), 'search', '--cand-path', './results/',
            '--injection-path', injection_path, '--only-injections', '--cutoff-frequency', '500']

    if ii is not None:
        command.extend(['--ii', str(ii)])

    subprocess.run(command)

def get_cand_data(ra, dec, date, ndays, inj_path, ii = None):
    inj_name = inj_path.split('/')[-1]
    if ii is not None:
        path = f'./results/injections/{ra}_{dec}_cumulative_power_spectra_stack_{date}_{ndays}_{inj_name}_({ii},)_candidates.npz'
    else:
        path = f'./results/injections/{ra}_{dec}_cumulative_power_spectra_stack_{date}_{ndays}_{inj_name}_candidates.npz'
    cands = np.load(path, allow_pickle = True)['candidate_dicts']
    return_list = []
    for cand in cands:
        temp_dict = {}
        temp_dict['freq'] = cand['freq']
        temp_dict['dm'] = cand['dm']
        temp_dict['sigma'] = cand['sigma']
        return_list.append(temp_dict)
    return return_list

def run_focus(ra, dec, date, ndays, path, focus, step = 1):

    injections = []

    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    i = 0
    while i < len(data):
        print(f'Starting injection {i+1}.')
        inj = data[i]
        call(path, ra, dec, ii = i)
        del inj['profile']
        cands = get_cand_data(ra, dec, date, ndays, path, ii = i)
        print(f'{len(cands)} candidates returned.')
        injections.append([inj, cands])
        i += step

    output = np.zeros((len(injections), 4))
    multicands = []

    for i in range(len(injections)):
        output[i, 0] = injections[i][0][focus]
        f = injections[i][0]['frequency']
        freq_closest = np.infty
        for cand in injections[i][1]:
            if np.abs(cand['freq'] - f) < np.abs(freq_closest - f):
                output[i, 1] = cand['freq']
                output[i, 2] = cand['dm']
                output[i, 3] = cand['sigma']
                freq_closest = cand['freq']

        if len(injections[i][1]) > 1:
            multicands.append([injections[i][0], len(injections[i][1])])

    return output, multicands
               

def populate(map_file, ra, dec, date, ndays, path):


    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    for i in range(len(data)):
        
        inj = data[i]
        print(f'Starting injection {i} with f = {inj["frequency"]}, DM = {inj["DM"]}, sigma = {inj["sigma"]}')
        call(path, ra, dec, ii = i)
        del inj['profile']
        cands = get_cand_data(ra, dec, date, ndays, path, ii = i)
        print(f'{len(cands)} candidates returned.')
        
        #grab the true candidate, not a harmonic...
        f = inj['frequency']
        freq_closest = np.infty

        if len(cands) != 0:

            for j in range(len(cands)):
                if np.abs(cands[j]['freq'] - f) < np.abs(freq_closest - f):
                    freq_closest = cands[j]['freq']
                    closest_idx = j

        with open(map_file, 'a') as file:
            
            file.write(f"{inj['frequency']} {inj['DM']}")
            
            if len(cands) != 0:
                file.write(f" {cands[closest_idx]['freq']} {cands[closest_idx]['dm']}")
                file.write(f" {cands[closest_idx]['sigma']/inj['sigma']}\n")
            
            else:
                file.write("0 0")
                file.write("0\n")
        print('Written to map.')
