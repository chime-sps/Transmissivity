import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
plt.rcParams.update({'font.size': 14})

def get_sensitivity(full_data, col, bins, gauss_dev, predicted = False):

    arr_injected, arr_retrieved = get_density1d(full_data, col, bins, predicted = predicted)

    arr_inj_smooth = gaussian_filter(arr_injected, gauss_dev)
    arr_ret_smooth = gaussian_filter(arr_retrieved, gauss_dev)

    arr_inj_p_smooth = arr_inj_smooth / np.sum(arr_inj_smooth)
    arr_ret_p_smooth = arr_ret_smooth / arr_inj_smooth

    arr_ret_p = arr_retrieved / arr_injected
    
    nan_mask = np.isnan(arr_ret_p)
    x = np.arange(len(arr_ret_p))
    arr_ret_p[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], arr_ret_p[~nan_mask])

    arr_ret_p_smooth[np.isnan(arr_ret_p_smooth)] = 0.

    return arr_inj_p_smooth, arr_ret_p, arr_ret_p_smooth

def get_density1d(full_data, col, bins, type = 'retrieved', predicted = False):

    injected_density, _ = np.histogram(full_data[:, col], bins = bins)

    if predicted:
        retrieved_mask = full_data[:, 5] > 6. #index is awkwardly hard-coded...
        retrieved_density, _ = np.histogram(full_data[retrieved_mask][:, col], 
                                              bins=bins)

        return injected_density, retrieved_density
        
    if type == 'retrieved':
        retrieved_mask = full_data[:, -1] > 0.
        retrieved_density, _ = np.histogram(full_data[retrieved_mask][:, col], 
                                              bins=bins)

        return injected_density, retrieved_density
    
    elif type == 'missed':
        sigma_mask = full_data[:, 8] > 6. #threshold for candidates 
        missed_mask = full_data[:, -1] < 0. 
        should_have_detected = full_data[sigma_mask & missed_mask]
        missed_density, _ = np.histogram(should_have_detected[:, col], 
                                              bins=bins)
        return injected_density, missed_density

def get_density2d(full_data, col1, col2, bins1, bins2, type = 'retrieved'):

    bin_edges = [bins1, bins2]

    cols = [col1, col2]

    injected_density, _ = np.histogramdd(full_data[:, cols], bins=bin_edges)

    if type == 'retrieved':
        retrieved_mask = full_data[:, -1] > 0.
        retrieved_density, _ = np.histogramdd(full_data[retrieved_mask][:, cols], 
                                              bins=bin_edges)

        return injected_density, retrieved_density
    
    elif type == 'missed':
        sigma_mask = full_data[:, 8] > 6. #threshold for candidates 
        missed_mask = full_data[:, -1] < 0. 
        should_have_detected = full_data[sigma_mask & missed_mask]
        missed_density, _ = np.histogramdd(should_have_detected[:, cols], 
                                              bins=bin_edges)
        return injected_density, missed_density

def make_levels(arr, logspace = False, Nlevels = 20):

    if logspace:
        return np.logspace(np.log10(arr[arr > 0].min()), np.log10(arr.max()), Nlevels)
    
    else:
        return np.linspace(0, 1, Nlevels)

def add_colorbar(fig, ax, cf, tick_locs):
    cbar = fig.colorbar(cf, ax=ax, location='right', pad=0.02)
    cbar.set_label('Probability', labelpad=10)
    cbar.set_ticks(tick_locs)
    return cbar

def add_crosshair(ax, x_bins, y_bins, x_dev, y_dev, x_log=False, y_log=False):

    x_vals = x_bins[:-1]
    y_vals = y_bins[:-1]

    if x_log:
        log_x = np.log(x_vals)
        x_center = np.exp(log_x[-1] - 0.12 * (log_x[-1] - log_x[0]))
        x_half_log = x_dev * (log_x[-1] - log_x[0]) / len(x_vals)
        x_lo = np.exp(np.log(x_center) - x_half_log)
        x_hi = np.exp(np.log(x_center) + x_half_log)
    else:
        x_center = x_vals[-1] - 0.12 * (x_vals[-1] - x_vals[0])
        x_half = x_dev * (x_vals[-1] - x_vals[0]) / len(x_vals)
        x_lo = x_center - x_half
        x_hi = x_center + x_half

    if y_log:
        log_y = np.log(y_vals)
        y_center = np.exp(log_y[-1] - 0.12 * (log_y[-1] - log_y[0]))
        y_half_log = y_dev * (log_y[-1] - log_y[0]) / len(y_vals)
        y_lo = np.exp(np.log(y_center) - y_half_log)
        y_hi = np.exp(np.log(y_center) + y_half_log)
    else:
        y_center = y_vals[-1] - 0.12 * (y_vals[-1] - y_vals[0])
        y_half = y_dev * (y_vals[-1] - y_vals[0]) / len(y_vals)
        y_lo = y_center - y_half
        y_hi = y_center + y_half

    kw = dict(color='white', linewidth=1.5, transform=ax.transData)
    ax.plot([x_lo, x_hi], [y_center, y_center], **kw)
    ax.plot([x_center, x_center], [y_lo, y_hi], **kw)
    ax.plot(x_center, y_center, 'w+', markersize=4)


