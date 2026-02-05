import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import click
from scipy.fft import fft, ifft

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command()
@click.argument(
        "filenames",
        nargs = -1,
        )
@click.option(
        "--sigma",
        default = 5,
        help = ("Sigma of the smoothing gaussian. Default is 5 bins."),
        )
@click.option(
        "--plot",
        is_flag = True,
        help = ("Plot the original and smoothed profiles."),
        )
@click.option(
        "--force",
        is_flag = True,
        help = ('Force every value outside 20 bins of peak to 0.'),
        )

def smooth(filenames, sigma, plot, force):
    for file in filenames:
        profile = np.load(file)
        smoothed = gaussian_filter1d(profile, sigma=sigma, mode='wrap')
        smoothed /= max(smoothed)

        if force:
            max_idx = np.argmax(smoothed)
            smoothed[:max_idx - 10] = 0
            smoothed[max_idx + 10:] = 0

        np.save(f'{file.rstrip(".npy")}_smooth', smoothed)
        

        if plot:
            plt.plot(profile)
            plt.plot(smoothed)
            plt.show()

if __name__ == "__main__":

    smooth()

