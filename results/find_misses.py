import numpy as np
import matplotlib.pyplot as plt

#RA, Dec, Date, TPA_idx, f, DM, flux, fwhm, predicted_sigma, predicted_nharm, output_sigma, output_nharm, output_f, output_dm
data = np.loadtxt('all_real_output.txt', skiprows = 1)

tpa_idx = []
with open('misses.txt', 'w') as f:
    for line in data:
        if line[8] > 6. and line[-1] < 0.:
            f.write(f'Pointing: {line[0]}, {line[1]}\nTPA_idx: {line[3]}\nf: {line[4]} Hz\nDM: {line[5]} pc / cm3\nS:{line[6]} mJy\nPredicted sigma: {line[8]}\n\n')
            tpa_idx.append(line[3])
            plt.plot(line[6], line[5], 'r.')
    unique_values, counts = np.unique(tpa_idx, return_counts=True)
    sort_idx = np.argsort(counts)[::-1]

    f.write('TPA_index    Count\n')
    f.write('-------------------\n')
    for i in range(len(sort_idx)):
        f.write(f'{unique_values[sort_idx[i]].astype("int")}    {counts[sort_idx[i]]}\n')

print(f'There were {len(tpa_idx)} missed injections.')
plt.xlabel('S (mJy)')
plt.xscale('log')
plt.ylabel(r'DM (pc cm$^{-3}$)')
plt.ylim(0, 100)
plt.title('Missed Injections')
plt.show()
