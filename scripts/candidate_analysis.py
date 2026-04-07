import numpy as np
import click

def process_file(cand_file_path, outfile):

    split_path = cand_file_path.split('_')
    ra = float(split_path[0])
    dec = float(split_path[1])
    days = float(split_path[6])

    cand_file = np.load(cand_file_path, allow_pickle = True)
    N_injections = len(cand_file['injection_dicts'])

    output = np.zeros((N_injections, 14))
    output[:, 0:3] = np.array([ra, dec, days])[np.newaxis, :]
    print(f'There were {N_injections} injections in this power spectrum.')

    cand_freqs = []
    cand_idx = []
    
    for i in range(cand_file['cand_count'].item()):
        cand = cand_file[f'candidate_{i}'].item()
        cand_freqs.append(cand['freq'])
        cand_idx.append(cand['injection_dict']['injection_index'])
   
    cand_freqs = np.asarray(cand_freqs)
    cand_idx = np.asarray(cand_idx)
    print(f'Candidate frequencies are {cand_freqs}.')
    print(f'Candidate indeces are {cand_idx}.')

    #using complicated if-tree to clarify logic and optimize slightly
    if N_injections == 1:
        inj = cand_file['injection_dicts'][0]
        output[0, 3:10] = (-1, inj['frequency'], inj['DM'], inj['flux'], inj['FWHM'], inj['detection_sigma'], inj['detection_nharm'])
        if len(cand_freqs) == 1:
            cand = cand_file['candidate_0'].item()
            print('One injection + one candidate --> matched!')
        elif len(cand_freqs) > 1:
            print(f'There are {len(cand_freqs)} candidates arising from one injection.')
            closest_match = np.argmin(np.abs(cand_freqs - inj['frequency']))
            print(f'Matched candidate with frequency {cand_freqs[closest_match]} to injection with frequency {inj["frequency"]}.')
            cand = cand_file[f'candidate_{closest_match}'].item()
        else:
            print('There are no candidates arising from this injection.')
            cand = None
        
        if cand is not None:
            output[0, 10:14] = (cand['sigma'], cand['features'].item()[3], cand['freq'], cand['dm'])
        else:
            output[0, 10:14] = -1 * np.ones(4)

    else:
        print('Multiple injections in this power spectrum --> evaluating potential candidate-injection matches.') 
        for i in range(N_injections):
            inj = cand_file['injection_dicts'][i]
            output[i, 3:10] = (-1, inj['frequency'], inj['DM'], inj['flux'], inj['FWHM'], inj['detection_sigma'], inj['detection_nharm'])
            print(f'Searching for candidates arising from injection {i}.') 
            if len(cand_freqs) == 0:
                print('There were no candidates retrieved from this power spectrum.')
                cand = None
            elif len(cand_freqs) > 0:
                same_injection = np.where(cand_idx == inj['injection_index'])[0]
                print(f'There are {len(same_injection)} candidates arising from this injection.')
                
                if len(same_injection) == 0:
                    print(f'Proceeding to next injection.')
                    cand = None

                else:
                    closest_match = np.argmin(np.abs(cand_freqs[same_injection] - inj['frequency']))
                    print(f'Matched candidate with frequency {cand_freqs[closest_match]} to injection with frequency {inj["frequency"]}.')
                    cand = cand_file[f'candidate_{closest_match}'].item()

            if cand is not None: 
                output[i, 10:14] = (cand['sigma'], cand['features'].item()[3], cand['freq'], cand['dm'])
            else:
                output[i, 10:14] = -1 * np.ones(4)
    
    with open(outfile, 'a') as f:
        for line in output:
            for item in line:
                f.write(f'{item} ')
            f.write('\n')

    return



@click.command()
@click.argument(
        "filenames",
        nargs = -1,
        )
@click.option(
        "--output", "-o",
        default = 'stack_injections.txt',
        )
def main(filenames, output):

    for filename in filenames:
    
        print(f'Processing {filename}.') 

        process_file(filename, output) 

if __name__ == "__main__":

    main()
