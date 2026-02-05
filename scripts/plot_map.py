from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def log_centers_to_edges(arr):
    """
    Convert center values to log-spaced edges.
    Sanitizes input to avoid log10 errors.
    """
    arr = np.asarray(arr)

    # Replace non-positive with smallest positive value in arr
    # Avoids log10(0) or log10(negative)
    positive_mask = arr > 0
    if not np.any(positive_mask):
        raise ValueError("All input values are <= 0, cannot compute log edges.")

    min_positive = np.min(arr[positive_mask])
    arr = np.where(arr > 0, arr, min_positive)

    log_arr = np.log10(arr)
    log_edges = np.zeros(len(arr) + 1)

    # Interior edges are midpoints in log space
    log_edges[1:-1] = 0.5 * (log_arr[1:] + log_arr[:-1])

    # Extrapolate first and last edges
    log_edges[0]  = log_arr[0]  - (log_edges[1] - log_arr[0])
    log_edges[-1] = log_arr[-1] + (log_arr[-1] - log_edges[-2])

    return 10**log_edges


def plot(output_pows, input_dms, input_freqs, scale=None, title=None):
    """
    Safe plotting function that:
    - Replaces non-finite output values with a small positive number
    - Avoids log10 errors
    - Ensures correct pcolormesh grid
    """

    # --------------------------
    # CLEAN THE OUTPUT
    # --------------------------
    output_pows = np.nan_to_num(output_pows, nan=0.0, posinf=0.0, neginf=0.0)

    # pcolormesh + LogNorm CANNOT handle zeros → replace 0 with tiny epsilon
    eps = np.max(output_pows) * 1e-12 if np.max(output_pows) > 0 else 1e-12
    output_pows = np.where(output_pows > 0, output_pows, eps)

    # --------------------------
    # LOG-SPACED EDGES
    # --------------------------
    dm_edges = log_centers_to_edges(input_dms)
    f_edges  = log_centers_to_edges(input_freqs)

    # --------------------------
    # BUILD MESHGRID OF EDGES
    # --------------------------
    DM, F = np.meshgrid(dm_edges, f_edges, indexing='ij')

    # --------------------------
    # PLOT
    # --------------------------
    plt.figure()
    mesh = plt.pcolormesh(F, DM, output_pows,
                          cmap='magma',
                          norm=LogNorm(),
                          shading='auto')

    plt.colorbar(mesh, label=r'Transmissivity (mJy$^{-1}$')
    plt.ylabel(r'DM (pc cm$^{-3}$)')
    plt.xlabel('f (Hz)')
    plt.title(title)

    if scale == 'log':
        plt.xscale('log')
        plt.yscale('log')

    plt.show()

