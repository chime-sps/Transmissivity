import numpy as np
import matplotlib.pyplot as plt

def log_histogram(data, xlabel, num_bins = 50, plot = False):
    
    log_min = np.log10(np.min(data))
    log_max = np.log10(np.max(data))
    bins = np.logspace(log_min, log_max, num_bins)

    if plot:
        plt.hist(data, bins=bins, edgecolor='black')
        plt.xscale('log')
        plt.xlabel(xlabel)
        plt.show()

    return bins

def local_mean_std(x, y, window=100):
    n = len(y)
    half = window // 2

    mean = np.empty(n)
    std = np.empty(n)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half)
        mean[i] = np.mean(y[lo:hi])
        std[i] = np.std(y[lo:hi])

    return mean, std

def local_mean_std_f(
    f,
    y,
    window,
    *,
    mode="linear"
):
    """
    Compute local mean and std of y using a sliding window
    defined in frequency space.

    Parameters
    ----------
    f : array
        Frequency array (must be sorted, positive for log mode).
    y : array
        Values to average.
    window : float
        Window size:
          - linear mode: absolute width in f
          - log mode: width in decades (dex)
    mode : {"linear", "log"}
        Windowing scheme.

    Returns
    -------
    mean, std : arrays
        Local mean and standard deviation at each f.
    """
    f = np.asarray(f)
    y = np.asarray(y)

    if mode not in {"linear", "log"}:
        raise ValueError("mode must be 'linear' or 'log'")

    if mode == "log":
        if np.any(f <= 0):
            raise ValueError("f must be positive for log windows")
        logf = np.log10(f)

    n = len(f)
    mean = np.empty(n)
    std = np.empty(n)

    for i in range(n):
        if mode == "linear":
            lo = f[i] - window / 2
            hi = f[i] + window / 2
            mask = (f >= lo) & (f <= hi)

        else:  # log mode
            lo = logf[i] - window / 2
            hi = logf[i] + window / 2
            mask = (logf >= lo) & (logf <= hi)

        yi = y[mask]

        mean[i] = np.mean(yi)
        std[i] = np.std(yi)

    return mean, std

def local_median_std_f(
    f,
    y,
    window,
    *,
    mode="linear"
):
    """
    Compute local mean and std of y using a sliding window
    defined in frequency space.

    Parameters
    ----------
    f : array
        Frequency array (must be sorted, positive for log mode).
    y : array
        Values to average.
    window : float
        Window size:
          - linear mode: absolute width in f
          - log mode: width in decades (dex)
    mode : {"linear", "log"}
        Windowing scheme.

    Returns
    -------
    mean, std : arrays
        Local mean and standard deviation at each f.
    """
    f = np.asarray(f)
    y = np.asarray(y)

    if mode not in {"linear", "log"}:
        raise ValueError("mode must be 'linear' or 'log'")

    if mode == "log":
        if np.any(f <= 0):
            raise ValueError("f must be positive for log windows")
        logf = np.log10(f)

    n = len(f)
    median = np.empty(n)
    std = np.empty(n)

    for i in range(n):
        if mode == "linear":
            lo = f[i] - window / 2
            hi = f[i] + window / 2
            mask = (f >= lo) & (f <= hi)

        else:  # log mode
            lo = logf[i] - window / 2
            hi = logf[i] + window / 2
            mask = (logf >= lo) & (logf <= hi)

        yi = y[mask]

        median[i] = np.median(yi)
        std[i] = np.std(yi)

    return median, std

