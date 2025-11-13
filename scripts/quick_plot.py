import numpy as np
import matplotlib.pyplot as plt
import corner

inj = np.load('timestream_pows_test.npz')
pows = inj['pows']
f_vals = inj['f']
dm_vals = inj['DM']
s_vals = inj['S']
d_vals = inj['delta']

f_slice = pows[:, 0, 0, 0]
print(f_vals, f_slice)
#plt.plot(f_vals[1:], f_slice[1:])
#plt.show()
exit()

# f_vals shape (N_f,)
# dm_vals shape (N_DM,)
# s_vals shape (N_S,)
# d_vals shape (N_delta,)

# Make coordinate grid for each dimension
F, DM, S, D = np.meshgrid(f_vals, dm_vals, s_vals, d_vals, indexing='ij')

# Flatten into vectors
samples = np.vstack([
    F.ravel(),
    DM.ravel(),
    S.ravel(),
    D.ravel()
]).T      # shape → (N_points, 4)

weights = pows.ravel()   # same length

# Corner plot
figure = corner.corner(
    samples,
    weights=weights,
    labels=[r"$f$", r"$DM$", r"$S$", r"$\Delta$"],
    show_titles=True
)

