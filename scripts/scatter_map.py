import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- Function to convert FWHM to marker size ---
def fwhm_to_markersize(fwhm, sizes, scale_factor=200, exponent=0.7):
    """
    Convert FWHM values to scatter marker sizes (pt^2) using power-law scaling.
    """
    norm_fwhm = (fwhm - np.min(sizes)) / (np.max(sizes) - np.min(sizes))
    marker_sizes = scale_factor * (norm_fwhm + 0.05)**exponent
    return marker_sizes

# --- Load data ---
inj = np.loadtxt('synthetic_map.txt')
freq = inj[:, 1]
dm   = inj[:, 2]
ratio = inj[:, 5] / inj[:, 3]
sizes = inj[:, 4]


# --- Convert FWHM to marker sizes ---
point_sizes = fwhm_to_markersize(sizes, sizes, scale_factor=200, exponent=0.7)

# --- Plotting ---
plt.figure(figsize=(8, 6))

# Increase general font sizes
plt.rcParams.update({
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 16
})

sc = plt.scatter(freq, dm,
                 c=ratio,
                 s=point_sizes,
                 cmap='magma',
                 norm=LogNorm(),
                 alpha=0.3,
                 edgecolors='black',
                 linewidths=0.3)

# --- Colorbar ---
cbar = plt.colorbar(sc)
cbar.set_label(r'Transmissivity (mJy$^{-1}$)', fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.xscale('log')
plt.xlabel('Frequency', fontsize=14)
plt.ylabel('DM', fontsize=14)
plt.title('Synthetic Map', fontsize=16)

# --- Size legend ---
example_FWHM = np.percentile(sizes, [10, 50, 90])
legend_sizes = fwhm_to_markersize(example_FWHM, sizes, scale_factor=200, exponent=0.7)

for f, s in zip(example_FWHM, legend_sizes):
    plt.scatter([], [], s=s, color='gray', edgecolor='black', alpha=0.7, label=f'FWHM = {f:.2f}')

plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title='Marker Size', fontsize=12, title_fontsize=12)

plt.tight_layout()
plt.show()

