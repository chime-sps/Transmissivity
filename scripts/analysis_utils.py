import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import LogFormatter
from matplotlib import colors
from scipy.ndimage import distance_transform_edt

def _column(data, col):
    """handle structured arrays"""
    if data.dtype.names is not None:
        return data[col]
    return data[:, col]

def fill_nans(arr):

    nan_mask = np.isnan(arr)
    if nan_mask.all() or not nan_mask.any():
        return arr
    if arr.ndim == 1:
        x = np.arange(len(arr))
        out = arr.copy()
        out[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], arr[~nan_mask])
        return out
    nearest = distance_transform_edt(nan_mask, return_distances = False, return_indices = True)
    return arr[tuple(nearest)]

def get_sensitivity(full_data, *P, stype = '1d', def_retrieval = 'all', predicted = False,
                    sigma_col = None, sigma_threshold = 6.,
                    apply_thin_mask = True, mask_cutoff = 20):

    if stype in ['1d', '1', '1D', 1]:
        col, bins, gauss_dev = P
        arr_injected, arr_retrieved = get_density1d(full_data, col, bins,
                                                    def_retrieval = def_retrieval,
                                                    predicted = predicted,
                                                    sigma_col = sigma_col,
                                                    sigma_threshold = sigma_threshold)
        beam_area = np.sqrt(2 * np.pi) * gauss_dev

    elif stype in ['2d', '2', '2D', 2]:
        col1, col2, bins1, bins2, dev1, dev2 = P
        gauss_dev = (dev1, dev2)
        arr_injected, arr_retrieved = get_density2d(full_data,
                                                    col1,
                                                    col2,
                                                    bins1,
                                                    bins2,
                                                    def_retrieval = def_retrieval,
                                                    predicted = predicted,
                                                    sigma_col = sigma_col,
                                                    sigma_threshold = sigma_threshold)
        beam_area = 2 * np.pi * dev1 * dev2
        
    else:
        print(f'stype was not a valid input.')
        return
    
    arr_inj_smooth = gaussian_filter(arr_injected.astype(float), gauss_dev)
    arr_ret_smooth = gaussian_filter(arr_retrieved.astype(float), gauss_dev)
    arr_inj_per_beam = arr_inj_smooth * beam_area

    # densities
    arr_ret_p = arr_retrieved / arr_injected
    arr_ret_p_smooth = arr_ret_smooth / arr_inj_smooth

    # mask thin
    if apply_thin_mask:
        thin = arr_inj_per_beam < mask_cutoff
        arr_ret_p = np.where(thin, np.nan, arr_ret_p)
        arr_ret_p_smooth = np.where(thin, np.nan, arr_ret_p_smooth)

    # interpolate
    arr_ret_p = fill_nans(arr_ret_p)
    arr_ret_p_smooth = fill_nans(arr_ret_p_smooth)

    return arr_inj_per_beam, arr_ret_p, arr_ret_p_smooth

def get_density1d(full_data, col, bins, dtype = 'retrieved', def_retrieval = 'all',
                  predicted = False, sigma_col = None, sigma_threshold = 6.):

    values = _column(full_data, col)
    injected_density, _ = np.histogram(values, bins = bins)

    if predicted:
        if sigma_col is None:
            raise ValueError("predicted=True needs sigma_col (index or field name of the predicted sigma).")
        retrieved_mask = _column(full_data, sigma_col) >= sigma_threshold
        retrieved_density, _ = np.histogram(values[retrieved_mask], bins=bins)
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
                  stype = 'retrieved', predicted = False, sigma_col = None,
                  sigma_threshold = 6.):

    bin_edges = [bins1, bins2]

    if predicted:
        if sigma_col is None:
            raise ValueError("predicted=True needs sigma_col (index or field name of the predicted sigma).")
        x = _column(full_data, col1)
        y = _column(full_data, col2)
        retrieved_mask = _column(full_data, sigma_col) >= sigma_threshold
        injected_density, _, _ = np.histogram2d(x, y, bins=bin_edges)
        retrieved_density, _, _ = np.histogram2d(x[retrieved_mask], y[retrieved_mask],
                                                 bins=bin_edges)
        return injected_density, retrieved_density

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
        abs_max = np.nanmax(np.abs(arr))
        levels = np.linspace(-abs_max, abs_max, Nlevels)
        return levels 
    
    elif logspace:
        return np.logspace(1., np.log10(arr.max()), Nlevels)
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

def make_contour_map(densities,
                     bins1, bins2, 
                     dev1, dev2, 
                     title,
                     cmap = 'magma',
                     ellipse_colors = ['w', 'w', 'w', 'w', 'w', 'w'],
                     colorbar_name = 'Probability of Retrieval',
                     cmap_diverging = False,
                     cmap_log = False,
                     log_ticks = None,
                     ):

    x_labels = ['Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)',
                'FWHM', 'FWHM', 'S (mJy)']
    y_labels = [r'DM (pc cm$^{-3}$)', 'S (mJy)', 'FWHM',
                'S (mJy)', r'DM (pc cm$^{-3}$)', r'DM (pc cm$^{-3}$)']
    
    xlogs = [True, True, True, True, True, True]
    ylogs = [False, True, True, True, False, False]

    # subtitles = ['Frequency vs DM', 'Frequency vs Flux', 'Frequency vs FWHM',
    #              'FWHM vs Flux', 'FWHM vs DM', 'Flux vs DM']

    fig, ax = plt.subplots(2, 3, figsize=(26, 14), layout = 'constrained')
    ax = ax.flatten()

    for i in range(6):

        cf = populate_contour_map(ax[i], 
                                  densities[i],
                                  bins1[i], bins2[i],
                                  dev1[i], dev2[i],
    #                              title = subtitles[i],
                                  xlabel = x_labels[i],
                                  ylabel = y_labels[i],
                                  xlog = xlogs[i],
                                  ylog = ylogs[i],
                                  cmap = cmap,
                                  ellipse_color = ellipse_colors[i],
                                  cmap_diverging = cmap_diverging,
                                  cmap_log = cmap_log,
                                  )

    cbar = fig.colorbar(cf, ax=ax, location='right', pad=0.08)
    cbar.set_label(colorbar_name, labelpad=10)

    if cmap_log:
        cbar.set_ticks(log_ticks)

    elif cmap_diverging:
        cbar.set_ticks(np.arange(-0.3, 0.4, 0.1))
    else:
        cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    fig.suptitle(title, 
                fontsize=40, fontweight='bold')

    plt.show()

def populate_contour_map(ax, 
                         density, 
                         bins1, bins2, 
                         dev1, dev2, 
                         title = None,
                         xlabel = None, ylabel = None,
                         xlog = False, ylog = False,
                         cmap = 'magma',
                         ellipse_color = 'w',
                         cmap_diverging = False,
                         cmap_log = False):
    levels = make_levels(density, 
                         logspace = cmap_log,
                         diverging = cmap_diverging,
                         Nlevels = 15) 
    
    if cmap_log:
        norm = colors.LogNorm(vmin=10, vmax=density.max())
    else:
        norm = None

    cf = ax.contourf(bins1[:-1], bins2[:-1], 
                         density, 
                         levels=levels,
                         cmap = cmap,
                         norm = norm)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_facecolor('k')

    if xlog:
        ax.set_xscale('log')

    if ylog:
        ax.set_yscale('log')

    ax.set_box_aspect(1)

    add_ellipse(ax, bins1, bins2, dev1, dev2, x_log=xlog, color = ellipse_color)

    return cf


def mask_thin(density, injections, min_inj = 1e2, stype = '1d', ):

    if stype in [1, '1', '1D', '1d']:
        return np.where(injections >= min_inj, density, np.nan)

    elif stype in [2, '2', '2D', '2d']:
        return [np.where(inj >= min_inj, m, np.nan) for m, inj in zip(density, injections)]

    else:
        print(f'[WARNING] {stype} not a valid dimension.')