import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

x = inj[:, 1]
y = inj[:, 2]
z = inj[:, 3] / inj[:, 5]

# Target grid for plotting (regular)
xi = np.linspace(min(x), max(x), 300)
yi = np.linspace(min(y), max(y), 300)
XI, YI = np.meshgrid(xi, yi)

# Interpolate
ZI = griddata((x, y), z, (XI, YI), method='linear')   # or 'nearest', 'cubic'

