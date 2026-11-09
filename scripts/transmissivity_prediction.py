import logging
import os
from datetime import datetime, timezone

import click
import numpy as np

import ps_processes.processes.ps_inject as ps_inject
from ps_processes.processes.ps_inject import Injection, TPA_profiles, kernels
from sps_common.constants import DM_CONSTANT, FREQ_BOTTOM, FREQ_TOP, TSAMP
from sps_common.interfaces.utilities import harmonic_sum

log = logging.getLogger(__name__)

PADDED_LENGTH = 1048576  
MAX_DM = 10**3.3  
RA = 50.814417767387674  
DEC = 39.90059501729749  
UTC_START = 1696587658.0  

# output columns of npz file
RESULT_DTYPE = np.dtype(
    [
        ("TPA_idx", np.int32),
        ("frequency", np.float64),
        ("DM", np.float64),  
        ("flux", np.float64),  
        ("FWHM", np.float64),  
        ("injected_nharm", np.int32),
        ("predicted_sigma", np.float64),
        ("predicted_nharm", np.int32),
    ]
)


class NoiseFreePowerSpectra:
    """
    Basically just exists to be a minimal pspec that we can read attributes from.
    """

    def __init__(
        self, tsamp, max_dm, ra, dec, date, ndays=1, padded_length=PADDED_LENGTH
    ):
        self.freq_labels = np.fft.rfftfreq(padded_length, d=tsamp)[:-1]

        disp_delay = 1 / FREQ_BOTTOM**2 - 1 / FREQ_TOP**2
        num_dms = int(DM_CONSTANT * max_dm * disp_delay / tsamp) + 1
        self.dms = np.arange(0, num_dms) / DM_CONSTANT / disp_delay * tsamp

        self.num_days = ndays
        self.power_spectra = np.broadcast_to(
            np.float32(self.num_days), (len(self.dms), len(self.freq_labels))
        )
        self.bad_freq_indices = [[]]
        self.ra = ra
        self.dec = dec
        self.datetimes = [date]


def make_full_harm_bins(freq_labels, num_harm=32):
    """
    From ps_search.py.
    """
    ps_length_search = (len(freq_labels) // num_harm) * num_harm
    return np.vstack(
        (
            np.arange(0, ps_length_search),
            harmonic_sum(num_harm, np.zeros(ps_length_search))[1],
        )
    ).astype(np.int32)


def frequency_choices(f_nyquist):
    
    f_dist = np.loadtxt(
        os.path.join(os.path.dirname(ps_inject.__file__), "atnf_freqs.txt"),
        usecols=[1],
    )
    f_log = np.logspace(-3, 2.7, int((4 / 6) * len(f_dist)))
    f_choices = np.concatenate([f_dist, f_log])
    return f_choices[f_choices < f_nyquist]


def disperse_true_dm_row(inj, prof_fft, kernels_dict):
    
    trial_dms = inj.trial_dms
    i_min = np.argmin(np.abs((inj.true_dm - 2 * inj.deltaDM) - trial_dms))
    i_max = np.argmin(np.abs((inj.true_dm + 2 * inj.deltaDM) - trial_dms))
    dm_indices = np.arange(i_min, i_max + 1)

    i0 = inj.true_dm_trial
    key = np.argmin(
        np.abs(np.abs((trial_dms[i0] - inj.true_dm) / inj.deltaDM) - kernels_dict["scaling"])
    )
    row = prof_fft * kernels_dict[str(key)][1 : len(prof_fft) + 1]
    return dm_indices, row


def predict_injection(pspec, full_harm_bins, kernels_dict, tsamp, f, dm, flux, profile):
    """
    Same as ps_inject() EXCEPT:
        - no red noise normalization
        - no RFI weighting
        - only disperse at true DM row
    """
    inj = Injection(pspec, full_harm_bins, dm, f, profile, flux=flux)

    freqs = pspec.freq_labels
    f_nyquist = freqs[-1]
    df = 1 / (tsamp * 2 * len(freqs))

    n_harm = min(32, int(np.floor(f_nyquist / f)))

    scaled_prof_fft, _ = inj.flux_to_power()
    if len(scaled_prof_fft) > n_harm:
        scaled_prof_fft = scaled_prof_fft[:n_harm]
    else:
        n_harm = len(scaled_prof_fft)

    windowed_prof_fft = inj.time_windowing(scaled_prof_fft)
    smeared_prof_fft = inj.smear_fft(windowed_prof_fft)
    dm_indices, true_dm_row = disperse_true_dm_row(inj, smeared_prof_fft, kernels_dict)

    true_dm_in_harms = inj.true_dm_trial - dm_indices[0]
    bins, true_row = inj.scalloping(true_dm_row, df)
    harms = np.zeros((len(dm_indices), len(true_row)))
    harms[true_dm_in_harms] = true_row

    _, predicted_nharm, predicted_sigma, _ = inj.predict_sigma(
        harms, bins, dm_indices, n_harm, True
    )

    return predicted_sigma, predicted_nharm, n_harm, inj.W * f


@click.command()
@click.option(
    "--tsamp",
    type=float,
    default=None,
    help="Sample time in microseconds. Default: 983 (CHAMPSS).",
)
@click.option(
    "-n", "--n-injections", type=int, default=10000, show_default=True,
    help="Number of injections to generate.",
)
@click.option(
    "-o", "--output", type=click.Path(), default=None,
    help="Output .npy file. Default: transmissivity_prediction_tsamp<tsamp>_ndays<ndays>.npy",
)
@click.option(
    "--ndays", type=click.IntRange(min=1), default=1, show_default=True,
    help="Number of days over which the injection is stacked.",
)
@click.option("--seed", type=int, default=None, help="Seed for the random generator.")
@click.option(
    "--max-dm", type=float, default=MAX_DM, show_default=True,
    help="Maximum trial DM.",
)
@click.option("--dm-min", type=float, default=1.0, show_default=True,
              help="Minimum trial DM.")
@click.option("--ra", type=float, default=RA, show_default=True, help="Pointing RA (deg), sets nchan.")
@click.option("--dec", type=float, default=DEC, show_default=True, help="Pointing Dec (deg), sets nchan.")
@click.option("--utc-start", type=float, default=UTC_START, show_default=True,
              help="Unix time of the observation for pointing map.")
@click.option("--s-min", type=float, default=5e-2, show_default=True, help="Minimum flux (mJy).")
@click.option("--s-max", type=float, default=5e-1, show_default=True, help="Maximum flux (mJy).")
def main(tsamp, n_injections, output, ndays, seed, max_dm, dm_min, ra, dec, utc_start, s_min, s_max):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger(ps_inject.__name__).setLevel(logging.WARNING)

    if tsamp is None:
        tsamp = TSAMP
    else:
        tsamp /= 1e6 #convert from input (us) to seconds
    ps_inject.TSAMP = tsamp

    if output is None:
        output = f"transmissivity_prediction_tsamp{tsamp:.8g}_ndays{ndays}.npy"

    rng = np.random.default_rng(seed)

    date = datetime.fromtimestamp(utc_start, tz=timezone.utc)
    pspec = NoiseFreePowerSpectra(tsamp, max_dm, ra, dec, date, ndays=ndays)
    f_nyquist = pspec.freq_labels[-1]
    log.info(
        f"tsamp = {tsamp * 1e6} us, ndays = {ndays}, Nyquist = {f_nyquist:.2f} Hz, "
        f"{len(pspec.dms)} DM trials up to {pspec.dms[-1]:.1f} pcc"
    )

    full_harm_bins = make_full_harm_bins(pspec.freq_labels)
    kernels_dict = {key: kernels[key] for key in kernels.files}

    f_choices = frequency_choices(f_nyquist)
    S_choices = np.logspace(np.log10(s_min), np.log10(s_max), 10000)
    n_profiles = len(TPA_profiles.files)
    dm_max = pspec.dms[-1]

    results = np.zeros(n_injections, dtype=RESULT_DTYPE)
    n_done = 0
    n_failed = 0

    try:
        for i in range(n_injections):
            f = rng.choice(f_choices)
            dm = rng.uniform(dm_min, dm_max)
            S = rng.choice(S_choices)
            prof_idx = int(rng.integers(n_profiles))
            profile = np.array(TPA_profiles[str(prof_idx)], dtype=np.float64)

            try:
                sigma, nharm, inj_nharm, fwhm = predict_injection(
                    pspec, full_harm_bins, kernels_dict, tsamp, f, dm, S, profile
                )
            except Exception as e:
                n_failed += 1
                log.debug(f"Injection {i} (f={f}, DM={dm}, S={S}) failed: {e!r}")
                continue

            results[n_done] = (prof_idx, f, dm, S, fwhm, inj_nharm, sigma, nharm)
            n_done += 1

            if (i + 1) % 500 == 0:
                log.info(f"{i + 1}/{n_injections} injections ({n_failed} failed)")
    finally:
        results = results[:n_done]
        np.save(output, results)
        log.info(
            f"Saved {n_done} injections to {output} "
            f"({n_failed} failed out of {n_done + n_failed} attempted)."
        )


if __name__ == "__main__":
    main()
