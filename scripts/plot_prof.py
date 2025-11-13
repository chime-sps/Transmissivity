import matplotlib.pyplot as plt
import numpy as np
import yaml

phi = np.linspace(0, 1, 1024)
file = 'high_snr_TPA_pulsars.npy'
profs = np.load(file)
plt.plot(phi, profs[50])
plt.show()
