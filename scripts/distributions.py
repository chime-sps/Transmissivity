import numpy as np
import numpy.random as rand


TPA_info = np.load('TPA_info.npz')


def get_frequencies(N):

    freqs = TPA_info['f']
    
    return rand.choice(freqs, size = N, replace = True)

def get_DMs(N, mean_DM = 30, DM_std = 10):

    gaussian = rand.normal(mean_DM, N - int(N/10), DM_std) 
    exponential = 

