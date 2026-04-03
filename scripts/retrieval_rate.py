import numpy as np
import click

def process_file(cand_file_path):

    split_path = cand_file_path.split('_')
    ra = float(split_path[0])
    dec = float(split_path[1])
    days = float(split_path[6])

    cand_file = np.load(cand_file_path, allow_pickle = True)
    N_injections = len(cand_file['injection_dicts'])

    output = np.zeros((N_injections, 14))
    output[:, 0:3] = np.array([ra, dec, days])[np.newaxis, :]
    
    for i in range(N_injections):
        
        inj = cand_file['injection_dicts'][i]
        output[i, 3:10] = (-1, inj['frequency'], inj['DM'], inj['flux'], inj['FWHM'], inj['detection_sigma'], inj['detection_nharm'])
        cand = look_for_injection(cand_file, inj)

        if cand is not None: 
            output[i, 10:14] = (cand['sigma'], cand['features'].item()[3], cand['freq'], cand['dm'])
        else:
            output[i, 10:14] = -1 * np.ones(4)

    else:
        output[i, 10:14] = -1 * np.ones(4)
    
    return output

def look_for_injection(cands, inj):

    cand_freqs = []
    cand_idx = []

    for i in range(cands['cand_count'].item()):
        cand = cands[f'candidate_{i}'].item()
        cand_freqs.append(cand['freq'])
        if cand['injection']:
            cand_idx.append(inj['injection_index'])
        else:
            cand_idx.append(-1)

    cand_freqs = np.asarray(cand_freqs)
    cand_idx = np.asarray(cand_idx)


    if len(cand_freqs) > 0:
        same_injection = np.where(cand_idx == i)[0]
        closest_match = np.argmin(np.abs(cand_freqs - inj['frequency']))
        if closest_match in same_injection:
            cand = cands[f'candidate_{closest_match}'].item()
            return cand

        else:
            return None

    else:
        return None

@click.command()
@click.argument(
        "filenames",
        nargs = -1,
        )

def main(filenames):

    all_output = []

    for filename in filenames:
    
        print(f'Processing {filename}.') 

        output = list(process_file(filename))
        all_output.extend(output)

    all_output = np.asarray(all_output)
    print(all_output.shape)
    print(f'Total fraction of recovered injections: {(100*len(all_output[all_output[:, -1] > 0.])/len(all_output)):.2f}%')
    print(f'Completeness fraction: {(100*len(all_output[(all_output[:, -1] > 0.) & (all_output[:, 8] > 6.)])/len(all_output)):.2f}%') 
if __name__ == "__main__":

    main()

