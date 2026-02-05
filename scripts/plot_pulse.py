import numpy as np
import matplotlib.pyplot as plt

profs = np.load('smoothed_baselined_TPA_pulses.npy')
plt.plot(profs[0])
plt.plot(profs[1])
plt.plot(profs[2])
plt.show()
exit()
