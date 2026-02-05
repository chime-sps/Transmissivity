import numpy as np

#RA	Dec	Date	TPA_idx	f		DM		S			FWHM-dependent quantity	predict_sigma	 
data = np.loadtxt('all_real_output.txt', skiprows = 1)
maxdm = np.loadtxt('maxdm_per_pointing.txt')

data = data[data[:, 2] == 20251113.0]
good_idx = []
for i in range(len(data)):
    #mask = (maxdm[:, 0] == data[i, 0]) & (maxdm[:, 1] == data[i, 1])
    mask = np.argmin((maxdm[:, 0] - data[i, 0])**2 + (maxdm[:, 1] - data[i, 1])**2)
    if data[i, 5] < maxdm[mask, 2]:
        good_idx.append(i)
    else:
        print(data[i, :2], data[i, 5], maxdm[mask])
        print('')
       
good_idx = np.array(good_idx)
TPA_idx = data[:, 3].astype('int')
fwhm_array = np.load('/home/squillace/Transmissivity/profiles/TPA_fwhm.npy')
fwhm = fwhm_array[TPA_idx]
data[:, 7] = fwhm[data[:, 3].astype('int')]
data[:, 2] = data[:, 2].astype('int')

with open('filtered_injections.txt', 'w') as f:
    for line in data:
        for item in line:
            f.write(f'{item} ')
        f.write('\n')
