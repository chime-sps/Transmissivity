import numpy as np
import matplotlib.pyplot as plt
from helpful_code import log_histogram, local_mean_std, local_mean_std_f, local_median_std_f

inj = np.loadtxt('../results/synthetic_map_with_tsky.txt')

f = inj[:, 1]
DM = inj[:, 2]
S = inj[:, 3]
FWHM = inj[:, 4]
sigma = inj[:, 5]
nharm = inj[:, 6]
T = sigma / S
T0 = np.median(T)

DM_bins = np.linspace(min(DM), max(DM), 10)
dDM = DM_bins[1] - DM_bins[0]
#f_bins = log_histogram(f, 'f (Hz)', num_bins = 20)
fig, ax = plt.subplots(figsize = (8, 5))
ax2 = ax.twinx()

cmap = plt.cm.winter
colors = cmap(np.linspace(0, 1, len(DM_bins) - 1))

yticks = []
yticklabels = []

for i in range(1, len(DM_bins)):
    mask = (DM > DM_bins[i - 1]) & (DM <= DM_bins[i]) 
    idx = np.argsort(f[mask])

    color = colors[i - 1]
    offset = 5*(i - 1)*T0
    offset = 0
    f_sorted = f[mask][idx]
    T_sorted = T[mask][idx]

    #T_mean, T_std = local_mean_std(f_sorted, T_sorted, window=1000)
    T_mean, T_std = local_mean_std_f(
                                    f_sorted,
                                    T_sorted,
                                    window=1.5,     
                                    mode="log")

    T_mean, T_std = local_median_std_f(
                                    f_sorted,
                                    T_sorted,
                                    window=1.5,     
                                    mode="log")
    # mean line
    ax.plot(
        f_sorted,
        T_mean + offset,
        color=color,
        lw=1.5,
        label=f'{np.median(DM[mask]):.1f}'
    )
    
    #contains useful info but borks the dynamic range...
    #ax.hlines(offset, xmin = f_sorted[0], xmax = f_sorted[-1], linestyle = '--', color = color)

    # std represented as line width (envelope)
    #ax.fill_between(
    #    f_sorted,
    #    T_mean - T_std + offset,
    #    T_mean + T_std + offset,
    #    color=color,
    #    alpha=0.3
    #)
    yticks.append(T0 + offset)
    yticklabels.append(f'{np.median(DM[mask]):.1f}')

ax.set_xlabel("f (Hz)")
#ax.set_yticks([])
ax.set_ylabel(r'Transmissivity (mJy$^{-1}$)')
ax.set_xscale('log')
ax.set_title('Waterfall Plot of Predicted Transmissivity Function')

ax.legend(title = r'Median DM (pc cm$^{-3}$)')
# Right-hand axis shows DM bins
#ax2.set_ylim(ax.get_ylim())
#ax2.set_yticks(yticks)
ax2.set_yticks([])
#ax2.set_yticklabels(yticklabels)
#ax2.set_ylabel(r"Median DM (pc cm$^{-3}$")

plt.show()
