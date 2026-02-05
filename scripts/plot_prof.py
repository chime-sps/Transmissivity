import matplotlib.pyplot as plt
import numpy as np
import yaml

#phi = np.linspace(0, 1, 1024)
file = 'B0154+61_profile_smooth.npy'
prof = np.load(file)
plt.plot(prof)
plt.show()
