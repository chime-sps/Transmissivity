import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def get_sensitivity(full_data, *P, stype = '1d', def_retrieval = 'all', predicted = False):

    if stype in ['1d', '1', '1D', 1]:
        col, bins, gauss_dev = P
        arr_injected, arr_retrieved = get_density1d(full_data,
                                                    col, 
                                                    bins, 
                                                    def_retrieval = def_retrieval,
                                                    predicted = predicted)
    elif stype in ['2d', '2', '2D', 2]:
        col1, col2, bins1, bins2, dev1, dev2 = P
        gauss_dev = (dev1, dev2)
        arr_injected, arr_retrieved = get_density2d(full_data,
                                                    col1,
                                                    col2,
                                                    bins1,
                                                    bins2,
                                                    def_retrieval = def_retrieval)
        
    else:
        print(f'stype was not a valid input.')
        return

    arr_inj_smooth = gaussian_filter(arr_injected, gauss_dev)
    arr_ret_smooth = gaussian_filter(arr_retrieved, gauss_dev)

    arr_inj_p_smooth = arr_inj_smooth / np.sum(arr_inj_smooth)
    arr_ret_p_smooth = arr_ret_smooth / arr_inj_smooth

    arr_ret_p = arr_retrieved / arr_injected
    
    shape = arr_ret_p.shape
    flat = arr_ret_p.flatten()
    nan_mask = np.isnan(flat)
    x = np.arange(len(flat))
    flat[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], flat[~nan_mask])
    arr_ret_p = flat.reshape(shape)

    arr_ret_p_smooth[np.isnan(arr_ret_p_smooth)] = 0.

    return arr_inj_p_smooth, arr_ret_p, arr_ret_p_smooth

def get_blindspots(full_data, *P, stype = '1d'):

    if stype in ['1d', '1', '1D', 1]:
        col, bins, gauss_dev = P
        arr_injected, arr_missed = get_density1d(full_data,
                                                    col, 
                                                    bins, 
                                                    stype = 'missed')
    elif stype in ['2d', '2', '2D', 2]:
        col1, col2, bins1, bins2, dev1, dev2 = P
        gauss_dev = (dev1, dev2)
        arr_injected, arr_missed = get_density2d(full_data,
                                                    col1,
                                                    col2,
                                                    bins1,
                                                    bins2,
                                                    stype = 'missed')
        
    else:
        print(f'stype was not a valid input.')
        return

    arr_inj_smooth = gaussian_filter(arr_injected, gauss_dev)
    arr_mis_smooth = gaussian_filter(arr_missed, gauss_dev)

    arr_inj_p_smooth = arr_inj_smooth / np.sum(arr_inj_smooth)
    arr_mis_p_smooth = arr_mis_smooth / arr_inj_smooth

    arr_mis_p = arr_missed / arr_injected
    
    shape = arr_mis_p.shape
    flat = arr_mis_p.flatten()
    nan_mask = np.isnan(flat)
    x = np.arange(len(flat))
    flat[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], flat[~nan_mask])
    arr_mis_p = flat.reshape(shape)

    arr_mis_p_smooth[np.isnan(arr_mis_p_smooth)] = 0.

    return arr_inj_p_smooth, arr_mis_p, arr_mis_p_smooth

def get_density1d(full_data, col, bins, dtype = 'retrieved', def_retrieval = 'all',
                  predicted = False):

    injected_density, _ = np.histogram(full_data[:, col], bins = bins)

    if predicted:
        retrieved_mask = full_data[:, 5] > 6. #index is awkwardly hard-coded...
        retrieved_density, _ = np.histogram(full_data[retrieved_mask][:, col], 
                                              bins=bins)

        return injected_density, retrieved_density
        
    if dtype == 'retrieved':
        if def_retrieval == 'all':
            retrieved_mask = full_data[:, -1] > 0.
        elif def_retrieval in ['first', '1', 1]:
            mask1 = full_data[:, -1] > 0
            mask2 = np.abs(full_data[:, -2] - full_data[:, 4]) / full_data[:, 4] < 0.05
            retrieved_mask = mask1 & mask2
        retrieved_density, _ = np.histogram(full_data[retrieved_mask][:, col], 
                                              bins=bins)

        return injected_density, retrieved_density
    
    elif dtype == 'missed':
        sigma_mask = full_data[:, 8] > 6. #threshold for candidates 
        missed_mask = full_data[:, -1] < 0. 
        should_have_detected = full_data[sigma_mask & missed_mask]
        missed_density, _ = np.histogram(should_have_detected[:, col], 
                                              bins=bins)
        return injected_density, missed_density

def get_density2d(full_data, col1, col2, bins1, bins2, def_retrieval = 'all',
                  stype = 'retrieved'):

    bin_edges = [bins1, bins2]

    cols = [col1, col2]

    if stype == 'retrieved':
        injected_density, _ = np.histogramdd(full_data[:, cols], bins=bin_edges)
        if def_retrieval == 'all':
            retrieved_mask = full_data[:, -1] > 0.
        elif def_retrieval in ['first', '1', 1]:
            mask1 = full_data[:, -1] > 0
            mask2 = np.abs(full_data[:, -2] - full_data[:, 4]) / full_data[:, 4] < 0.05
            retrieved_mask = mask1 & mask2
        retrieved_density, _ = np.histogramdd(full_data[retrieved_mask][:, cols], 
                                              bins=bin_edges)

        return injected_density, retrieved_density
    
    elif stype == 'missed':
        sigma_mask = full_data[:, 8] > 6. #threshold for candidates 
        missed_mask = full_data[:, -1] < 0. 
        should_have_detected = full_data[sigma_mask & missed_mask]
        injected_density, _ = np.histogramdd(full_data[sigma_mask][:, cols], 
                                             bins=bin_edges)
        missed_density, _ = np.histogramdd(should_have_detected[:, cols], 
                                              bins=bin_edges)
        return injected_density, missed_density

def make_levels(arr, logspace = False, Nlevels = 20, diverging = False):

    if diverging: 
        #cannot accomodate logspace
        abs_max = np.max(np.abs(arr))
        levels = np.linspace(-abs_max, abs_max, Nlevels)
        return levels 
    
    elif logspace:
        return np.logspace(np.log10(arr[arr > 0].min()), np.log10(arr.max()), Nlevels)
    
    else:
        return np.linspace(0, 1, Nlevels)

def add_colorbar(fig, ax, cf, tick_locs):
    cbar = fig.colorbar(cf, ax=ax, location='right', pad=0.02)
    cbar.set_label('Probability', labelpad=10)
    cbar.set_ticks(tick_locs)
    return cbar

def add_ellipse(ax, x_bins, y_bins, x_dev, y_dev, x_log=False, y_log=False, color = 'white'):
    n_x = len(x_bins) - 1
    n_y = len(y_bins) - 1

    rx_ax = x_dev / n_x
    ry_ax = y_dev / n_y

    cx_ax = 1.0 - 0.05 - rx_ax
    cy_ax = 1.0 - 0.05 - ry_ax

    t = np.linspace(0, 2 * np.pi, 300)
    x_ellipse_ax = cx_ax + rx_ax * np.cos(t)
    y_ellipse_ax = cy_ax + ry_ax * np.sin(t)

    ax.plot(x_ellipse_ax, y_ellipse_ax,
            color=color, linewidth=1.5,
            transform=ax.transAxes)
    
def add_crosshair(ax, x_bins, x_dev):
    n_x = len(x_bins) - 1
    rx_ax = x_dev / n_x
    cx_ax = 0.05 + rx_ax
    cy_ax = 1.0 - 0.05 - 0.02  # room for T-caps

    style = dict(color='black', linewidth=1.5, transform=ax.transAxes)

    ax.plot([cx_ax - rx_ax, cx_ax + rx_ax], [cy_ax, cy_ax], **style)
    ax.plot([cx_ax - rx_ax, cx_ax - rx_ax], [cy_ax - 0.02, cy_ax + 0.02], **style)
    ax.plot([cx_ax + rx_ax, cx_ax + rx_ax], [cy_ax - 0.02, cy_ax + 0.02], **style)