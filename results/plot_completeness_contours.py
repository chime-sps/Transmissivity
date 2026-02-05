import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 14})

# Load and filter data
data = np.loadtxt('all_real_output.txt')
#data = data[data[:, 2] == 20251113.0 | data[:, 2] == 20251116.0]
data = data[np.isin(data[:, 2], [20251113.0, 20251116.0])]
# Print summary statistics
print(f'Number of pointings: {len(np.unique(data[:, 0:2], axis=0))}')
print(f'Number of days: {len(np.unique(data[:, 2]))}')
print(f'Total fraction of recovered injections: {(100*len(data[data[:, -1] > 0.])/len(data)):.2f}%')

# Extract parameters
f = data[:, 4]
DM = data[:, 5]
S = data[:, 6]
TPA_idx = data[:, 3].astype('int')
fwhm_array = np.load('/home/squillace/Transmissivity/profiles/TPA_fwhm.npy')
fwhm = fwhm_array[TPA_idx]

# Define binning parameters
N_bins = 15  # Number of bins for each parameter

# Create bins for each parameter
f_bins = np.logspace(np.log10(min(f)), np.log10(max(f)), N_bins + 1)
DM_bins = np.linspace(min(DM), max(DM), N_bins + 1)
S_bins = np.logspace(np.log10(min(S)), np.log10(max(S)), N_bins + 1)
fwhm_bins = np.linspace(min(fwhm), max(fwhm), N_bins + 1)

# Function to calculate completeness for a pair of parameters
def calculate_completeness(param1, param2, bins1, bins2):
    """
    Calculate completeness in 2D parameter space.
    Completeness = percentage of data where output_sigma > 0
    """
    completeness = np.zeros((len(bins1) - 1, len(bins2) - 1))

    for i in range(len(bins1) - 1):
        mask1 = (param1 > bins1[i]) & (param1 <= bins1[i + 1])

        for j in range(len(bins2) - 1):
            mask2 = (param2 > bins2[j]) & (param2 <= bins2[j + 1])
            mask = mask1 & mask2

            cell_elements = data[mask]

            if len(cell_elements) > 0:
                yes = cell_elements[cell_elements[:, -1] > 0.]
                retrieval_rate = len(yes) / len(cell_elements)
                completeness[i, j] = retrieval_rate
            else:
                completeness[i, j] = np.nan

    return completeness

# Parameter information
params = {
    'f': (f, f_bins, 'Frequency (Hz)', True),
    'DM': (DM, DM_bins, 'DM (pc cm$^{-3}$)', False),
    'S': (S, S_bins, 'Flux Density (mJy)', True),
    'fwhm': (fwhm, fwhm_bins, 'FWHM', False)
}

param_names = ['f', 'DM', 'S', 'fwhm']

# Create corner plot
print("Creating corner plot...")
fig = plt.figure(figsize=(8, 8))
gs = GridSpec(3, 3, figure=fig, hspace=0.15, wspace=0.15, 
              left=0.15, right=0.88, bottom=0.12, top=0.93)

# Manually define subplot positions for the 6 plots in lower triangle
subplot_positions = {
    (1, 0): (0, 0),  # DM vs f
    (2, 0): (1, 0),  # S vs f
    (2, 1): (1, 1),  # S vs DM
    (3, 0): (2, 0),  # fwhm vs f
    (3, 1): (2, 1),  # fwhm vs DM
    (3, 2): (2, 2),  # fwhm vs S
}

plot_count = 0
for i, param_name1 in enumerate(param_names):
    for j, param_name2 in enumerate(param_names):

        # Only plot lower triangle (i > j)
        if i > j:
            plot_count += 1
            print(f"  Creating subplot {plot_count}/6: {param_name1} vs {param_name2}")
            grid_pos = subplot_positions[(i, j)]
            ax = fig.add_subplot(gs[grid_pos[0], grid_pos[1]])

            param1, bins1, label1, log1 = params[param_name1]
            param2, bins2, label2, log2 = params[param_name2]

            # Calculate completeness
            completeness = calculate_completeness(param1, param2, bins1, bins2)

            # Create mesh for plotting using bin centers
            bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
            bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
            X, Y = np.meshgrid(bin_centers2, bin_centers1)

            # Plot contours
            levels = np.linspace(0, 1, 11)
            contourf = ax.contourf(X, Y, completeness, levels=levels,
                                   cmap='magma', extend='both')
            contour = ax.contour(X, Y, completeness, levels=levels[::2],
                                colors='white', linewidths=0.5, alpha=0.5)

            # Set scales
            if log2:
                ax.set_xscale('log')
            if log1:
                ax.set_yscale('log')

            # Labels
            if j == 0:  # Leftmost column
                ax.set_ylabel(label1)
            else:
                ax.set_yticklabels([])

            if i == 3:  # Bottom row
                ax.set_xlabel(label2)
            else:
                ax.set_xticklabels([])

            # Set axis limits
            ax.set_xlim(bins2[0], bins2[-1])
            ax.set_ylim(bins1[0], bins1[-1])

# Add colorbar
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
cbar = fig.colorbar(contourf, cax=cbar_ax)
cbar.set_label('Completeness', rotation=270, labelpad=25, fontsize=16)

plt.suptitle('Completeness Plots for 13/11/2026', fontsize=18, y=0.97)
plt.savefig('completeness_corner_plot.png', dpi=300, bbox_inches='tight')
print("\nSaved: completeness_corner_plot.png")

# Create individual plots for better detail
print("\nCreating individual plots...")
param_pairs = [
    ('f', 'DM'),
    ('f', 'S'),
    ('f', 'fwhm'),
    ('DM', 'S'),
    ('DM', 'fwhm'),
    ('S', 'fwhm')
]

fig2, axes = plt.subplots(2, 3, figsize=(8, 8))
axes = axes.flatten()

for idx, (param_name1, param_name2) in enumerate(param_pairs):
    print(f"  Creating plot {idx+1}/6: {param_name1} vs {param_name2}")
    ax = axes[idx]

    param1, bins1, label1, log1 = params[param_name1]
    param2, bins2, label2, log2 = params[param_name2]

    # Calculate completeness
    completeness = calculate_completeness(param1, param2, bins1, bins2)

    # Create mesh for plotting using bin centers
    bin_centers2 = (bins2[:-1] + bins2[1:]) / 2
    bin_centers1 = (bins1[:-1] + bins1[1:]) / 2
    X, Y = np.meshgrid(bin_centers2, bin_centers1)

    # Plot contours
    levels = np.linspace(0, 1, 11)
    contourf = ax.contourf(X, Y, completeness, levels=levels,
                           cmap='magma', extend='both')
    contour = ax.contour(X, Y, completeness, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                        colors='white', linewidths=1, alpha=0.7)
    ax.clabel(contour, inline=True, fontsize=10, fmt='%.1f')

    # Set scales
    if log2:
        ax.set_xscale('log')
    if log1:
        ax.set_yscale('log')

    # Labels
    ax.set_xlabel(label2)
    ax.set_ylabel(label1)

    # Set axis limits
    ax.set_xlim(bins2[0], bins2[-1])
    ax.set_ylim(bins1[0], bins1[-1])

# Add unified colorbar on the right
fig2.subplots_adjust(right=0.9, left=0.1)
cbar_ax2 = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
cbar2 = fig2.colorbar(contourf, cax=cbar_ax2)
cbar2.set_label('Completeness', rotation=270, labelpad=20, fontsize=14)

plt.savefig('completeness_individual_plots.png', dpi=300, bbox_inches='tight')
print("Saved: completeness_individual_plots.png")

print("\n" + "="*50)
print("All plots completed successfully!")
print("="*50)

# Uncomment the line below if you want to display plots interactively
plt.show()
