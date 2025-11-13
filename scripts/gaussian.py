import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-0.5*((x - mu)/sig)**2)/(sig*np.sqrt(2*np.pi))

x = np.linspace(0, 1, 1024)
profs = np.zeros((50, 1024))
sigmas = np.logspace(-2, -1)

for i in range(50):
    profs[i] = gaussian(x, 0.5, sigmas[i])

np.save('duty_cycle.npy', profs)
