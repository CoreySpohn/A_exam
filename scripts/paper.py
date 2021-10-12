"""
This module is intended to recreate all of the figures presented in the paper
"""
import functools
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import ffmpeg
import functions as fun
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.time import Time
from matplotlib.colors import ListedColormap
from planet import Planet
from planet_population import PlanetPopulation
from scipy import signal
from scipy.stats import gaussian_kde
from tqdm import tqdm


def time_vs_prob_det_both_failures(data_path):
    """
    This will create the plot showing both failures in the prob det vs time plot
    """
    # Planet parameters
    a = 2 * u.AU
    e = 0
    w_p = 0 * u.rad
    W = 0 * u.rad
    I = 1.82 * u.rad
    Mp = 16.30 * u.M_earth
    Rp = 3.5 * u.R_earth
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    # Telescope properties
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    #### Testing failure modes
    a = 2 * u.AU
    e = 0
    w_p = 0 * u.rad
    W = 0 * u.rad
    I = 1.65 * u.rad
    Mp = 15.86 * u.M_earth
    Rp = 3.46 * u.R_earth
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    # Telescope properties
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5
    rv_error = 1.25 * u.m / u.s

    # RV times and error
    initial_rv_time = Time(2000.0, format="decimalyear")
    # rv_error = 1.25 * u.m / u.s

    # Imaging times
    initial_obs_time = Time(2010.0, format="decimalyear")
    final_obs_time = Time(2030, format="decimalyear")
    timestep = 1 * u.d
    img_times = Time(
        np.arange(initial_obs_time.jd, final_obs_time.jd, step=timestep.to("d").value),
        format="jd",
    )

    scatter_kwargs = {
        "color": "lime",
        "alpha": 1,
        "s": 250,
        "marker": "P",
        "label": "True planet",
        "edgecolors": "black",
        "linewidth": 2,
    }
    plot_kwargs = {
        "color": "lime",
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "dashdot",
    }
    true_inputs = {
        "a": a,
        "e": e,
        "W": W,
        "I": I,
        "w": w_p,
        "Mp": Mp,
        "Rp": Rp,
        "f_sed": [3],
        "p": p,
        "M0": RV_M0,
        "t0": initial_rv_time,
        "rv_error": rv_error,
    }
    planet = Planet(
        "Planet",
        dist,
        Ms,
        [1],
        plot_kwargs,
        scatter_kwargs,
        keplerian_inputs=true_inputs,
    )
    planet.gen_base_path(rv_error, initial_rv_time, initial_obs_time, IWA, OWA, dMag0)

    telescope = [IWA, OWA, dMag0]

    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.95, 5))

    save_path = Path("plots/failure_plots/prob_det_vs_time_both_failure.pdf")
    failure_mode_calculations(
        data_path,
        planet,
        img_times,
        telescope,
        [],
        [],
        create_prob_det_plots=True,
        custom_save_path=save_path,
        colors=colors,
    )


def failure_mode_calculations(
    data_path,
    planet,
    img_times,
    telescope_params,
    failure_rates,
    first_if_list,
    create_prob_det_plots=False,
    colors=None,
    method_suffix="",
    custom_save_path=None,
    return_failure_times=False
):
    """
    This function does the calculations for failure modes, taking in a planet object and the necessary information and returning the mode of failure and the times of failure

    """

    # These are used for the intermittent failure metric
    days_for_intermittent_failure = 1 * u.d
    intermittent_failure_threshold = 0.95
    intermittent_failure_slope = 0.0001
    intermittent_failure_threshold_test = 0.9
    testing_min_max_diff = True

    dispersion_failure_prominence = 0.05

    # Getting input information
    IWA = telescope_params[0]
    OWA = telescope_params[1]
    dMag0 = telescope_params[2]

    timestep = (img_times[1] - img_times[0]).jd * u.d

    # Planet information
    ratio = planet.rv_error.decompose().value / planet.K.decompose().value
    # Create path for the planet that includes the visibility calculations
    planet_path = Path(
        data_path,
        "planet",
        str(planet.base_path)
        + f"_imgtimei{img_times[0].decimalyear:.0f}_imgtimef{img_times[-1].decimalyear:.0f}_imgtimestep{(img_times[1].decimalyear*u.yr-img_times[0].decimalyear*u.yr).to(u.d):.1f}".replace(
            " ", ""
        )
        + ".p",
    )
    if planet_path.exists():
        loaded_planet_visibility = True
        with open(planet_path, "rb") as f:
            planet = pickle.load(f)
    else:
        loaded_planet_visibility = False
    # Create path for the RV fits
    rv_fits_path = Path(
        data_path,
        "planet_population",
        str(planet.base_path)
        + f"_imgtimei{img_times[0].decimalyear:.0f}_imgtimef{img_times[-1].decimalyear:.0f}_imgtimestep{(img_times[1].decimalyear*u.yr-img_times[0].decimalyear*u.yr).to(u.d):.1f}_".replace(
            " ", ""
        )
        + method_suffix
        + ".p",
    )
    with open(rv_fits_path, "rb") as f:
        rv_fits = pickle.load(f)

    # Calculate planet visibility times if they are not already loaded
    if not loaded_planet_visibility:
        planet.visibility_times = []
        for t in img_times:
            WA_t, dMag_t = planet.prop_for_imaging(t)
            is_visible = (IWA < WA_t) & (OWA > WA_t) & (dMag0 > dMag_t)
            planet.visibility_times.append(is_visible)
        with open(planet_path, "wb") as f:
            pickle.dump(planet, f)

    # Flag to tell whether a planet is always detectable/visible
    always_visible = min(planet.visibility_times) == 1
    # if min(planet.visibility_times) == 1:
    # always_visible = True
    # # always_visible_inclinations.append(i)
    # else:
    # always_visible = False
    # not_always_visible_inclinations.append(i)

    rv_fits.above_threshold = []
    rv_fits.failure_times = []
    rv_fits.failure_time_indices = []
    rv_fits.consecutive_fails = []

    # Find the max values for the signal and the times of them
    maxima, _ = signal.find_peaks(
        rv_fits.percent_detectable,
        distance=20,
        prominence=dispersion_failure_prominence,
    )
    # Go through the maxima and remove redundant ones that represent the same peak
    maxima_to_remove = []
    for i, val in enumerate(maxima[0:-1]):
        # Get all the times between it and the next maxima
        intermediary_inds = np.arange(val, maxima[i + 1], 1)
        intermediary_vals = np.array(rv_fits.percent_detectable)[intermediary_inds]
        # If the value stays constant at 1, store the current peak to remove
        if min(intermediary_vals) == 1:
            maxima_to_remove.append(i)
    maxima = np.delete(maxima, maxima_to_remove)

    # Getting the minima
    inverted = 1 - np.array(rv_fits.percent_detectable)
    minima, _ = signal.find_peaks(
        inverted, distance=20, prominence=dispersion_failure_prominence
    )

    # Do the intermittent failure calculations
    if len(minima) > 0 and len(maxima) > 0:
        # Initialize the counters
        intermittent_failure = False

        # Create the windows for this planet's percent detectable curve
        windows = np.array([], dtype=int)
        # Figure out whether the first peak is a max or a min
        if maxima[0] < minima[0]:
            windows = np.append(windows, 0)

        # Add all the minima as they are how I'm defining the windows
        windows = np.append(windows, minima)

        # Now count the number of peaks in each window
        num_peaks_in_windows, _ = np.histogram(maxima, windows)

        # For each part of the window get the max value and the threshold value
        window_thresholds = np.array([], dtype=int)
        for i, window_start in enumerate(windows):
            if i == len(windows) - 1:
                break
            window_end = windows[i + 1]
            if num_peaks_in_windows[i] == 0:
                # Don't look if a maxima wasn't detected in this range
                window_thresholds = np.append(window_thresholds, None)
                continue
            relevant_probabilities = rv_fits.percent_detectable[window_start:window_end]
            # Just finding the maximum value in case there are more than one
            # peaks, might have some weird edge cases but I haven't seen any
            # worth noting
            max_prob = max(relevant_probabilities)
            if testing_min_max_diff:
                min_prob = min(relevant_probabilities)
                threshold = (
                    intermittent_failure_threshold_test * (max_prob - min_prob)
                    + min_prob
                )
            else:
                threshold = intermittent_failure_threshold * max_prob
            # threshold = intermittent_failure_threshold*max_prob
            window_thresholds = np.append(window_thresholds, threshold)

        # Now find the times where it is above the relevant threshold and save
        # them to an array
        intermittent_failure_inds = []
        intermittent_failure_vals = []
        intermittent_failure_counter = 0
        for i, percent in enumerate(rv_fits.percent_detectable):
            # Go until we hit the start of the first window
            if i < windows[0]:
                continue
            if i >= windows[-1]:
                break
            # Get the current window and threshold
            i_window = np.digitize(i, windows) - 1  # digitize starts at 1
            i_threshold = window_thresholds[i_window]
            if i_threshold is None:
                continue
            next_percent = rv_fits.percent_detectable[i + 1]
            i_slope = (next_percent - percent) / (timestep.to(u.d).value)

            # Check if it's making a bad prediction
            if (
                (percent >= i_threshold)
                and (planet.visibility_times[i] == 0)
                and (abs(i_slope) < intermittent_failure_slope)
            ):
                intermittent_failure_counter += 1
                if (
                    intermittent_failure_counter * timestep
                    > days_for_intermittent_failure
                ):
                    intermittent_failure = True
                    intermittent_failure_inds.append(i)
                    intermittent_failure_vals.append(percent)
            else:
                intermittent_failure_counter = 0

        # Now calculate the times for the intermittent failures
        if intermittent_failure:
            intermittent_failure_times = img_times.decimalyear[
                intermittent_failure_inds
            ]

            # Save the time of first intermittent failure along with the ratio
            first_if_list.append(
                (
                    round(ratio, 2),
                    intermittent_failure_times[0] - img_times.decimalyear[0],
                )
            )

            # Get the ratio of intermittent failures versus the total time
            # planet is not detectable
            total_not_detectable = len(img_times) - sum(planet.visibility_times)[0]
            intermittent_failure_rate = (
                len(intermittent_failure_inds) / total_not_detectable
            )
            failure_rates.append((round(ratio, 2), intermittent_failure_rate))
        else:
            intermittent_failure_times = None
            intermittent_failure_vals = None

        # Now look for dispersion failure
        end_of_last_window = windows[-1]
        max_before_simulation_end = maxima[-1] > end_of_last_window
        if max_before_simulation_end:
            time_from_last_extrema = img_times.decimalyear[maxima[-1]]
            dispersion_start = maxima[-1]
        else:
            time_from_last_extrema = img_times.decimalyear[end_of_last_window]
            dispersion_start = end_of_last_window
        ending_time = img_times.decimalyear[-1] - time_from_last_extrema
        if ending_time > planet.T.to(u.yr).value:
            dispersion_failure = True
            dispersion_failure_times = img_times.decimalyear[
                dispersion_start : len(img_times) - 1
            ]
            dispersion_failure_vals = rv_fits.percent_detectable[
                dispersion_start : len(img_times) - 1
            ]
            # first_df_list.append((round(ratio, 2), dispersion_failure_times[0] - img_times.decimalyear[0]))
        else:
            dispersion_failure = False
            dispersion_failure_times = None
            dispersion_failure_vals = None
    else:
        intermittent_failure = False
        intermittent_failure_times = None
        intermittent_failure_vals = None
        dispersion_failure = True
        dispersion_failure_times = img_times.decimalyear
        dispersion_failure_vals = rv_fits.percent_detectable

    if create_prob_det_plots:
        fig_det, ax_det = plt.subplots(figsize=[8, 4])
        # plot the base prob_det vs time curve
        ax_det.plot(
            img_times.decimalyear,
            planet.visibility_times,
            label="True planet",
            linestyle="--",
            color=colors[0],
            zorder=0,
        )
        ax_det.plot(
            img_times.decimalyear,
            rv_fits.percent_detectable,
            label="Multivariate Gaussian",
            color=colors[4],
            zorder=4,
        )
        # Temporary tests for peak prominences
        # ax_det.scatter(
        # img_times[maxima].decimalyear,
        # np.array(rv_fits.percent_detectable)[maxima],
        # label="Maxima",
        # marker="x",
        # )
        # prominences = signal.peak_prominences(rv_fits.percent_detectable, maxima)[0]
        # contour_heights = np.array(rv_fits.percent_detectable)[maxima] - prominences
        # ax_det.vlines(
        # x=img_times[maxima].decimalyear,
        # ymin=contour_heights,
        # ymax=np.array(rv_fits.percent_detectable)[maxima],
        # )
        # Adding the windows
        # ax_det.vlines(x=img_times[windows].decimalyear, ymin=0, ymax=1)
        # ax_det.scatter(img_times[windows].decimalyear, np.array(rv_fits.percent_detectable)[windows], label='Window', marker='+', s=100)
        # breakpoint()

    # breakpoint()
    # Determine the failure mode
    if dispersion_failure and intermittent_failure:
        failure_mode = "both_failure"
        if create_prob_det_plots:
            ax_det.scatter(
                dispersion_failure_times,
                dispersion_failure_vals,
                marker="d",
                label="Dispersion failure",
                color=colors[2],
                zorder=3,
            )
            ax_det.scatter(
                intermittent_failure_times,
                intermittent_failure_vals,
                marker="v",
                label="Intermittent failures",
                color=colors[3],
                zorder=3,
            )
            ax_det.set_title(f"Both failures. Ratio {ratio:.2f}")

    elif dispersion_failure and not intermittent_failure:
        failure_mode = "dispersion_failure"
        if create_prob_det_plots:
            ax_det.scatter(
                dispersion_failure_times,
                dispersion_failure_vals,
                marker="d",
                label="Dispersion failure",
                color=colors[2],
                zorder=3,
            )
            ax_det.set_title(f"Dispersion failure. Ratio {ratio:.2f}")
    elif not dispersion_failure and intermittent_failure:
        failure_mode = "intermittent_failure"
        if create_prob_det_plots:
            ax_det.set_title(f"Intermittent failure. Ratio {ratio:.2f}")
            ax_det.scatter(
                intermittent_failure_times,
                intermittent_failure_vals,
                marker="v",
                label="Intermittent failures",
                color=colors[3],
                zorder=3,
            )
    elif always_visible:
        failure_mode = "no_failure_(always_detectable)"
    else:
        failure_mode = "no_failure_(not_always_detectable)"
        if create_prob_det_plots:
            ax_det.set_title(f"No failure. Ratio {ratio:.2f}")

    # Final plot saving adjusting and saving
    if create_prob_det_plots:
        ax_det.set_yticks(np.linspace(0, 1, 6))
        ax_det.set_xlabel("Time (years)")
        ax_det.set_ylabel("Probability of detection")
        ax_det.legend(loc="lower left", framealpha=1)
        fig_det.tight_layout()
        if custom_save_path:
            fig_det.savefig(custom_save_path, dpi=300)
        else:
            save_path = Path(
                "plots/failure_plots", f"{ratio:.2f}", failure_mode, planet.base_path
            ).with_suffix(".pdf")
            fig_det.savefig(save_path, dpi=300)
        plt.close(fig_det)
    if return_failure_times:
        return failure_mode, failure_rates, first_if_list, intermittent_failure_times, intermittent_failure_vals, dispersion_failure_times, dispersion_failure_vals
    else:
        return failure_mode, failure_rates, first_if_list


def error_in_time_comparisons(data_path):
    """
    This function creates a plot showing how different errors evolve in time.
    It creates a series of images with the largest error as a base layer and
    the smaller errors overlaying one another in the s vs dMag space
    """
    a = 2 * u.AU
    e = 0
    # i = 90*u.deg
    i = 100 * u.deg
    fixed_K_val = 1 * u.m / u.s
    w_p = 0 * u.rad
    W = 0 * u.rad
    Ms = 1 * u.M_sun
    p = 0.37
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    T = 2 * np.pi * np.sqrt(a ** 3 / (Ms * const.G))
    Mp = (
        fixed_K_val
        * (Ms) ** (2 / 3)
        * np.sqrt(1 - e ** 2)
        / np.sin(i.to(u.rad))
        * (T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    # Number of consistent orbits to generate
    n_fits = 50000

    # RV observation times
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)
    param_error = 0.1  # Also m/s
    rv_errors = [0.1, 0.5, 1] * u.m / u.s

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    planet_list = []
    orbit_list = []
    for rv_error in rv_errors:
        # Creating the planet instance
        scatter_kwargs = {
            "color": "red",
            "alpha": 1,
            "s": 2,
            "marker": "o",
            "label": "True planet",
            "edgecolors": "black",
            "linewidth": 0.2,
        }
        plot_kwargs = {
            "color": "lime",
            "alpha": 1,
            "linewidth": 1,
            "label": "True planet",
            "linestyle": "dashdot",
        }
        # The inputs for the planet class
        Rp = Planet.RfromM(None, Mp.to(u.M_earth).value)[0]
        true_inputs = {
            "a": a,
            "e": e,
            "W": W,
            "I": i,
            "w": w_p,
            "Mp": Mp,
            "Rp": Rp,
            "f_sed": [3],
            "p": p,
            "M0": RV_M0,
            "t0": initial_rv_time,
            "rv_error": rv_error,
        }
        planet = Planet(
            "Planet",
            dist,
            Ms,
            [1],
            plot_kwargs,
            scatter_kwargs,
            keplerian_inputs=true_inputs,
        )

        base_path = planet.gen_base_path(
            rv_error, initial_rv_time, final_rv_time, IWA, OWA, dMag0
        )
        rv_curve_path = Path(data_path, "rv_curve", str(base_path) + ".csv")
        post_path = Path(data_path, "post", str(base_path) + ".p")
        chains_path = Path(data_path, "chains", str(base_path) + ".csv")
        orbits_path = Path(data_path, "planet_population", str(base_path) + ".p")

        # Generate the chains and RV curve
        if rv_curve_path.exists():
            # print(rv_curve_path)
            rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
        else:
            rv_df = planet.simulate_rv_observations(rv_times, rv_error)
            rv_df.to_csv(rv_curve_path)

        if chains_path.exists():
            chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
        else:
            params = fun.gen_radvel_params(planet, param_error)
            post = fun.gen_posterior(rv_df, params, param_error)
            chains = fun.gen_chains(post)
            with open(post_path, "wb") as f:
                pickle.dump(post, f)
            chains.to_csv(chains_path)
        planet_list.append(planet)

        # Create the orbit fits
        orbit_fit_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }

        orbit_scatter_kwargs = {
            "color": "white",
            "alpha": 0.1,
            "s": 0.01,
            "label": "RV fit cloud",
        }
        orbit_plot_kwargs = {
            "color": "white",
            "alpha": 1,
            "label": "RV fit cloud",
            "linewidth": 1,
            "linestyle": "-",
        }
        if orbits_path.exists():
            with open(orbits_path, "rb") as f:
                orbits = pickle.load(f)
        else:
            orbits = PlanetPopulation(
                "RV fits",
                dist,
                Ms,
                [1],
                orbit_plot_kwargs,
                orbit_scatter_kwargs,
                chains=chains,
                options=orbit_fit_options,
            )
            with open(orbits_path, "wb") as f:
                pickle.dump(orbits, f)
        orbit_list.append(orbits)

    period_nums = [0, 4, 8]
    # period_nums = [0, 1]
    plot_times = Time(
        [
            final_rv_time.decimalyear + period * T.to(u.yr).value
            for period in period_nums
        ],
        format="decimalyear",
    )
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, periods+1), format='decimalyear')
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, (periods*2+1)), format='decimalyear')
    # plot_times = Time([2010, 2020], format='decimalyear')
    fig, axes = plt.subplots(nrows=len(plot_times), ncols=len(rv_errors))
    time_plot_path = Path(
        "plots",
        "dispersion_in_time_overlaid",
        f"dispersion_in_time_i{i.to(u.deg).value:.0f}.pdf",
    )
    cols = [
        f"Ratio: {ratio.value}" for ratio in rv_errors / fixed_K_val.decompose().value
    ]
    rows = [
        f"Period: {period}\nYear: {year:.0f}"
        for period, year in zip(period_nums, plot_times.decimalyear)
    ]
    pad = 5
    scatter = False

    # Color map for the density plot
    cmap = plt.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    # set alpha for last element
    # my_cmap[:,-1] = np.linsspace(0,1,cmap.N)
    # my_cmap[:128,-1] = np.zeros(128)
    # my_cmap[:32,-1] = np.zeros(32)
    my_cmap[:32, -1] = np.zeros(32)
    my_cmap = ListedColormap(my_cmap)
    cmap_scale = [None, None]
    temp_scale = [
        0,
        4575,
    ]  # Using this so that it only loops once, print cmap_scale to get the correct values
    # levels = np.logspace(-4, 0, 13)
    levels = np.logspace(-5, 0, 16)
    for i, time_i in enumerate(plot_times):
        # Get the true planet's location
        for j, error in enumerate(rv_errors):
            ax = axes[i, j]
            planet = planet_list[j]
            s_p, dMag_p = planet.prop_for_imaging(time_i)
            planet_separation = s_p.to(u.arcsec)
            orbits = orbit_list[j]
            s, dMag = orbits.prop_for_imaging(time_i)
            if scatter:
                ax.scatter(
                    s.to(u.arcsec).value,
                    dMag,
                    label=f"Error of {error.decompose().value:.2f} m/s",
                    s=0.1,
                    alpha=0.2,
                )
            else:
                separation_for_plot = s.to(u.arcsec)
                xx, yy = np.mgrid[
                    0.9
                    * min(separation_for_plot) : 1.1
                    * max(separation_for_plot) : 100j,
                    0.9 * min(dMag) : 1.1 * max(dMag) : 100j,
                ]
                # xx, yy = np.mgrid[fs_ax.get_xlim()[0]:fs_ax.get_xlim()[1]:50j, fs_ax.get_ylim()[0]:fs_ax.get_ylim()[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([separation_for_plot.value, dMag])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                f_max = np.amax(f)
                f_min = np.amin(f)
                if cmap_scale[0] is None or cmap_scale[0] > f_min:
                    cmap_scale[0] = f_min
                if cmap_scale[1] is None or cmap_scale[1] < f_max:
                    cmap_scale[1] = f_max

                f_scaled = f / temp_scale[1]
                # print(f'Scaled f: {np.amax(f_scaled)}\nUnscaled f: {np.amax(f)}')
                cfset = ax.contourf(
                    xx, yy, f_scaled, levels=levels, norm=mpl.colors.LogNorm()
                )
                # cfset = ax.contourf(xx, yy, f_scaled, locator=mpl.ticker.LogLocator(), cmap=cmap)
                # fig.colorbar(cfset, ax=ax)
            # ax.set_title(f"Appearance after {time.decimalyear - final_rv_time.decimalyear:.1f} years")
            if ax.is_first_row():
                ax.annotate(
                    cols[j],
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                )
            if ax.is_last_row():
                ax.set_xticks(np.arange(0, 0.45, 0.1))
                ax.set_xlabel(r"$\alpha$ (arcsec)")
            else:
                ax.set_xticks([])

            if ax.is_first_col():
                ax.annotate(
                    rows[i],
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    size="small",
                )
                ax.set_yticks(np.arange(18, 34, 4))
                ax.set_ylabel(r"$\Delta_{mag}$")
            else:
                ax.set_yticks([])

            # Plot the actual planet's location
            ax.scatter(
                planet_separation,
                dMag_p,
                **planet.scatter_kwargs,
            )
            ax.set_xlim([0, 0.3])
            ax.set_ylim([18, 30])
        # ax.plot([IWA.to(u.arcsec).value, OWA.to(u.arcsec).value], [dMag0, dMag0], linewidth=0.5, color='black')
        # ax.plot([IWA.to(u.arcsec).value, IWA.to(u.arcsec).value], [0, dMag0], linewidth=0.5, color='black')
    # fig.legend(markerscale=20)
    # fig.colorbar(cfset)
    fig.subplots_adjust(left=0.2, top=0.95)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cfset, cax=cbar_ax)
    cbar.set_ticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0])
    cbar.set_label(r"Normalized Density (arcsec $^{-1}$ $\Delta_{mag}$ $^{-1}$)")
    fig.savefig(time_plot_path, dpi=300)
    # print(cmap_scale)


def consistent_orbits_error_comparisons(data_path):
    """
    This creates the pdf comparison of the construction method for various error values
    """
    a = 2 * u.AU
    e = 0
    # i = 90*u.deg
    i = 100 * u.deg
    fixed_K_val = 1 * u.m / u.s
    w_p = 0 * u.rad
    W = 0 * u.rad
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    T = 2 * np.pi * np.sqrt(a ** 3 / (Ms * const.G))
    Mp = (
        fixed_K_val
        * (Ms) ** (2 / 3)
        * np.sqrt(1 - e ** 2)
        / np.sin(i.to(u.rad))
        * (T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    Rp = Planet.RfromM(None, Mp.to(u.M_earth).value)[0]
    # Number of consistent orbits to generate
    n_fits = 50000

    # RV observation times
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)
    param_error = 0.1  # Also m/s
    rv_errors = [0.1, 0.5, 1] * u.m / u.s

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    ##########
    # Colors and kwargs
    ##########
    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.95, 5))

    scatter_kwargs = {
        "color": "red",
        "alpha": 1,
        "s": 2,
        "marker": "o",
        "label": "True planet",
        "edgecolors": "black",
        "linewidth": 0.2,
    }
    plot_kwargs = {
        "color": "lime",
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "dashdot",
    }
    mg_scatter_kwargs = {
        "color": "white",
        "alpha": 0.1,
        "s": 0.01,
        "label": "RV fit cloud",
    }
    mg_plot_kwargs = {
        "color": "white",
        "alpha": 1,
        "label": "RV fit cloud",
        "linewidth": 1,
        "linestyle": "-",
    }
    # Credible interval kwargs
    ci_scatter_kwargs = {
        "color": "blue",
        "alpha": 1,
        "s": 10,
        "label": "Credible interval",
        "marker": ">",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    ci_plot_kwargs = {
        "color": colors[3],
        "alpha": 1,
        "label": "Credible interval",
        "markersize": 5,
        # "linestyle": "dashed",
        "linewidth": 1,
    }
    # Maximum likelihood kwargs
    ml_scatter_kwargs = {
        "color": "brown",
        "alpha": 1,
        "s": 10,
        "label": "Max likelihood",
        "marker": "x",
        "edgecolors": "white",
    }
    ml_plot_kwargs = {
        "color": colors[2],
        "alpha": 1,
        "label": "Max likelihood",
        "markersize": 5,
        # "linestyle": "dotted",
        "linewidth": 1,
    }
    # Kernel density estimate kwargs
    kde_scatter_kwargs = {
        "color": "green",
        "alpha": 1,
        "s": 10,
        "label": "Max KDE",
        "marker": "*",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    kde_plot_kwargs = {
        "color": colors[1],
        "alpha": 1,
        "label": "Max KDE",
        "markersize": 5,
        "linewidth": 1,
        # "linestyle": (0, (5,1)),# "linestyle": (0, (5, 10)),
    }

    planet_list, mg_list, ml_list, ci_list, kde_list = [], [], [], [], []
    for rv_error in rv_errors:
        # Creating the planet populations
        # Creating the planet instance
        # The inputs for the planet class
        true_inputs = {
            "a": a,
            "e": e,
            "W": W,
            "I": i,
            "w": w_p,
            "Mp": Mp,
            "Rp": Rp,
            "f_sed": [3],
            "p": p,
            "M0": RV_M0,
            "t0": initial_rv_time,
            "rv_error": rv_error,
        }
        planet = Planet(
            "Planet",
            dist,
            Ms,
            [1],
            plot_kwargs,
            scatter_kwargs,
            keplerian_inputs=true_inputs,
        )
        planet_list.append(planet)

        base_path = planet.gen_base_path(
            rv_error, initial_rv_time, final_rv_time, IWA, OWA, dMag0
        )
        rv_curve_path = Path(data_path, "rv_curve", str(base_path) + ".csv")
        post_path = Path(data_path, "post", str(base_path) + ".p")
        chains_path = Path(data_path, "chains", str(base_path) + ".csv")
        mg_path = Path(data_path, "planet_population", str(base_path) + ".p")
        ml_path = Path(data_path, "planet_population", str(base_path) + "_ml.p")
        ci_path = Path(data_path, "planet_population", str(base_path) + "_ci.p")
        kde_path = Path(data_path, "planet_population", str(base_path) + "_kde.p")

        # Generate the chains and RV curve
        if rv_curve_path.exists():
            # print(rv_curve_path)
            rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
        else:
            rv_df = planet.simulate_rv_observations(rv_times, rv_error)
            rv_df.to_csv(rv_curve_path)

        if chains_path.exists():
            chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
        else:
            params = fun.gen_radvel_params(planet, param_error)
            post = fun.gen_posterior(rv_df, params, param_error)
            chains = fun.gen_chains(post)
            with open(post_path, "wb") as f:
                pickle.dump(post, f)
            chains.to_csv(chains_path)

        # Create the orbit fits
        mg_fit_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        if mg_path.exists():
            with open(mg_path, "rb") as f:
                mg = pickle.load(f)
        else:
            mg = PlanetPopulation(
                "Multivariate Gaussian",
                dist,
                Ms,
                [1],
                mg_plot_kwargs,
                mg_scatter_kwargs,
                chains=chains,
                options=mg_fit_options,
            )
            with open(mg_path, "wb") as f:
                pickle.dump(mg, f)

        mg_list.append(mg)
        # Credible interval planet population
        ci_chain = chains.quantile([0.5])
        ci_planet_inputs = {
            "period": ci_chain["per1"].values[0] * u.d,
            "secosw": ci_chain["secosw1"].values[0],
            "sesinw": ci_chain["sesinw1"].values[0],
            "K": ci_chain["k1"].values[0],
            "T_c": Time(ci_chain["tc1"].values[0], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }

        ci_planet = Planet(
            "Credible interval",
            dist,
            Ms,
            [1],
            ci_plot_kwargs,
            ci_scatter_kwargs,
            rv_inputs=ci_planet_inputs,
        )

        ci_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
        # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
        ci_std = chains.std()
        if ci_path.exists():
            with open(ci_path, "rb") as f:
                ci = pickle.load(f)
        else:
            ci = PlanetPopulation(
                "Credible interval population",
                dist,
                Ms,
                [1],
                ci_plot_kwargs,
                ci_scatter_kwargs,
                options=ci_pp_options,
                base_planet=ci_planet,
                base_planet_errors=ci_std,
            )
            with open(ci_path, "wb") as f:
                pickle.dump(ci, f)
        ci_list.append(ci)
        # Find the maximimum likelihood planet
        ml_chain = chains.loc[chains["lnprobability"].idxmax()]
        ml_planet_inputs = {
            "period": ml_chain["per1"] * u.d,
            "secosw": ml_chain["secosw1"],
            "sesinw": ml_chain["sesinw1"],
            "K": ml_chain["k1"],
            "T_c": Time(ml_chain["tc1"], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        ml_planet = Planet(
            "Max likelihood",
            dist,
            Ms,
            [1],
            ml_plot_kwargs,
            ml_scatter_kwargs,
            rv_inputs=ml_planet_inputs,
        )
        ml_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        if ml_path.exists():
            with open(ml_path, "rb") as f:
                ml = pickle.load(f)
        else:
            ml = PlanetPopulation(
                "Max likelihood",
                dist,
                Ms,
                [1],
                ml_plot_kwargs,
                ml_scatter_kwargs,
                options=ml_pp_options,
                base_planet=ml_planet,
            )
            with open(ml_path, "wb") as f:
                pickle.dump(ml, f)
        ml_list.append(ml)

        # Find the gaussian kde planet
        kde = mg.get_kde_estimate(chains)
        kde_vals = kde.x
        kde_planet_inputs = {
            "period": kde_vals[0] * u.d,
            "T_c": Time(kde_vals[1], format="jd"),
            "secosw": kde_vals[2],
            "sesinw": kde_vals[3],
            "K": kde_vals[4],
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        kde_planet = Planet(
            "Max kernel density estimate",
            dist,
            Ms,
            [1],
            kde_plot_kwargs,
            kde_scatter_kwargs,
            rv_inputs=kde_planet_inputs,
        )
        kde_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        if kde_path.exists():
            with open(kde_path, "rb") as f:
                kde = pickle.load(f)
        else:
            kde = PlanetPopulation(
                "Kernel density estimate",
                dist,
                Ms,
                [1],
                kde_plot_kwargs,
                kde_scatter_kwargs,
                options=kde_pp_options,
                base_planet=kde_planet,
            )
            with open(kde_path, "wb") as f:
                pickle.dump(kde, f)
        kde_list.append(kde)
        # vals = ml.e
        # print(f"median: {np.median(vals)}\nstd: {np.std(vals)}\n")

    pop_list = [ml_list, kde_list, ci_list, mg_list]
    rows = ["Max likelihood", "Max KDE", "Credible\ninterval", "Multivariate\nGaussian"]
    # period_nums = [0, 4, 8]
    # period_nums = [0, 1]
    # plot_times = Time([final_rv_time.decimalyear + period*T.to(u.yr).value for period in period_nums], format='decimalyear')
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, periods+1), format='decimalyear')
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, (periods*2+1)), format='decimalyear')
    # plot_times = Time([2010, 2020], format='decimalyear')
    fig, axes = plt.subplots(nrows=len(pop_list), ncols=len(rv_errors))
    construction_method_vs_error_path = Path(
        "plots",
        "construction_method_vs_error",
        f"construction_method_vs_error_{i.to(u.deg).value:.0f}.pdf",
    )
    cols = [
        f"Ratio: {ratio.value}" for ratio in rv_errors / fixed_K_val.decompose().value
    ]
    # rows = [f'Period: {period}\nYear: {year:.0f}' for period, year in zip( period_nums, plot_times.decimalyear )]
    pad = 5
    scatter = False

    # Color map for the density plot
    cmap = plt.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    # set alpha for last element
    # my_cmap[:,-1] = np.linsspace(0,1,cmap.N)
    # my_cmap[:128,-1] = np.zeros(128)
    # my_cmap[:32,-1] = np.zeros(32)
    my_cmap[:32, -1] = np.zeros(32)
    my_cmap = ListedColormap(my_cmap)
    cmap_scale = [None, None]
    temp_scale = [
        0,
        4575,
    ]  # Using this so that it only loops once, print cmap_scale to get the correct values
    levels = np.logspace(-5, 0, 16)
    start_time = Time(2010, format="decimalyear")
    for i, pop in enumerate(pop_list):
        # Get the true planet's location
        for j, error in enumerate(rv_errors):
            ax = axes[i, j]
            planet = planet_list[j]
            s_p, dMag_p = planet.prop_for_imaging(start_time)
            planet_separation = s_p.to(u.arcsec)
            population = pop[j]
            s, dMag = population.prop_for_imaging(start_time)
            if scatter:
                ax.scatter(
                    s.to(u.arcsec).value,
                    dMag,
                    label=f"Error of {error.decompose().value:.2f} m/s",
                    s=0.1,
                    alpha=0.2,
                )
            else:
                separation_for_plot = s.to(u.arcsec)
                xx, yy = np.mgrid[
                    0.9
                    * min(separation_for_plot) : 1.1
                    * max(separation_for_plot) : 100j,
                    0.9 * min(dMag) : 1.1 * max(dMag) : 100j,
                ]
                # xx, yy = np.mgrid[fs_ax.get_xlim()[0]:fs_ax.get_xlim()[1]:50j, fs_ax.get_ylim()[0]:fs_ax.get_ylim()[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([separation_for_plot.value, dMag])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                f_max = np.amax(f)
                f_min = np.amin(f)
                if cmap_scale[0] is None or cmap_scale[0] > f_min:
                    cmap_scale[0] = f_min
                if cmap_scale[1] is None or cmap_scale[1] < f_max:
                    cmap_scale[1] = f_max

                f_scaled = f / temp_scale[1]
                # print(f'Scaled f: {np.amax(f_scaled)}\nUnscaled f: {np.amax(f)}')
                cfset = ax.contourf(
                    xx, yy, f_scaled, levels=levels, norm=mpl.colors.LogNorm()
                )
                # cfset = ax.contourf(xx, yy, f_scaled, locator=mpl.ticker.LogLocator(), cmap=cmap)
                # fig.colorbar(cfset, ax=ax)
            # ax.set_title(f"Appearance after {time.decimalyear - final_rv_time.decimalyear:.1f} years")
            if ax.is_first_row():
                ax.annotate(
                    cols[j],
                    xy=(0.5, 1),
                    xytext=(0, pad),
                    xycoords="axes fraction",
                    textcoords="offset points",
                    ha="center",
                    va="baseline",
                )
            if ax.is_last_row():
                ax.set_xticks(np.arange(0, 0.45, 0.1))
                ax.set_xlabel(r"$\alpha$ (arcsec)")
            else:
                ax.set_xticks([])

            if ax.is_first_col():
                ax.annotate(
                    rows[i],
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    size="small",
                    rotation=90,
                )
                ax.set_yticks(np.arange(18, 34, 4))
                ax.set_ylabel(r"$\Delta_{mag}$")
            else:
                ax.set_yticks([])

            # Plot the actual planet's location
            ax.scatter(
                planet_separation,
                dMag_p,
                **planet.scatter_kwargs,
            )
            ax.set_xlim([0, 0.3])
            ax.set_ylim([18, 30])
    fig.subplots_adjust(left=0.2, top=0.95)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cfset, cax=cbar_ax)
    cbar.set_ticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0])
    cbar.set_label(r"Normalized Density (arcsec $^{-1}$ $\Delta_{mag}$ $^{-1}$)")
    fig.savefig(construction_method_vs_error_path, dpi=300)
    print(cmap_scale)


def consistent_orbits_error_comparisons_video(data_path):
    """
    This animates the construction method vs error ratio pdf plot
    """
    a = 2 * u.AU
    e = 0
    # i = 90*u.deg
    i = 100 * u.deg
    fixed_K_val = 1 * u.m / u.s
    w_p = 0 * u.rad
    W = 0 * u.rad
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    T = 2 * np.pi * np.sqrt(a ** 3 / (Ms * const.G))
    Mp = (
        fixed_K_val
        * (Ms) ** (2 / 3)
        * np.sqrt(1 - e ** 2)
        / np.sin(i.to(u.rad))
        * (T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    Rp = Planet.RfromM(None, Mp.to(u.M_earth).value)[0]
    # Number of consistent orbits to generate
    n_fits = 50000

    # RV observation times
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)
    param_error = 0.1  # Also m/s
    rv_errors = [0.1, 0.5, 1] * u.m / u.s

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    ##########
    # Colors and kwargs
    ##########
    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.95, 5))

    scatter_kwargs = {
        "color": "red",
        "alpha": 1,
        "s": 2,
        "marker": "o",
        "label": "True planet",
        "edgecolors": "black",
        "linewidth": 0.2,
    }
    plot_kwargs = {
        "color": "lime",
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "dashdot",
    }
    mg_scatter_kwargs = {
        "color": "white",
        "alpha": 0.1,
        "s": 0.01,
        "label": "RV fit cloud",
    }
    mg_plot_kwargs = {
        "color": "white",
        "alpha": 1,
        "label": "RV fit cloud",
        "linewidth": 1,
        "linestyle": "-",
    }
    # Credible interval kwargs
    ci_scatter_kwargs = {
        "color": "blue",
        "alpha": 1,
        "s": 10,
        "label": "Credible interval",
        "marker": ">",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    ci_plot_kwargs = {
        "color": colors[3],
        "alpha": 1,
        "label": "Credible interval",
        "markersize": 5,
        # "linestyle": "dashed",
        "linewidth": 1,
    }
    # Maximum likelihood kwargs
    ml_scatter_kwargs = {
        "color": "brown",
        "alpha": 1,
        "s": 10,
        "label": "Max likelihood",
        "marker": "x",
        "edgecolors": "white",
    }
    ml_plot_kwargs = {
        "color": colors[2],
        "alpha": 1,
        "label": "Max likelihood",
        "markersize": 5,
        # "linestyle": "dotted",
        "linewidth": 1,
    }
    # Kernel density estimate kwargs
    kde_scatter_kwargs = {
        "color": "green",
        "alpha": 1,
        "s": 10,
        "label": "Max KDE",
        "marker": "*",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    kde_plot_kwargs = {
        "color": colors[1],
        "alpha": 1,
        "label": "Max KDE",
        "markersize": 5,
        "linewidth": 1,
        # "linestyle": (0, (5,1)),# "linestyle": (0, (5, 10)),
    }

    planet_list, mg_list, ml_list, ci_list, kde_list = [], [], [], [], []
    for rv_error in rv_errors:
        # Creating the planet populations
        # Creating the planet instance
        # The inputs for the planet class
        true_inputs = {
            "a": a,
            "e": e,
            "W": W,
            "I": i,
            "w": w_p,
            "Mp": Mp,
            "Rp": Rp,
            "f_sed": [3],
            "p": p,
            "M0": RV_M0,
            "t0": initial_rv_time,
            "rv_error": rv_error,
        }
        planet = Planet(
            "Planet",
            dist,
            Ms,
            [1],
            plot_kwargs,
            scatter_kwargs,
            keplerian_inputs=true_inputs,
        )
        planet_list.append(planet)

        base_path = planet.gen_base_path(
            rv_error, initial_rv_time, final_rv_time, IWA, OWA, dMag0
        )
        rv_curve_path = Path(data_path, "rv_curve", str(base_path) + ".csv")
        post_path = Path(data_path, "post", str(base_path) + ".p")
        chains_path = Path(data_path, "chains", str(base_path) + ".csv")
        mg_path = Path(data_path, "planet_population", str(base_path) + ".p")
        ml_path = Path(data_path, "planet_population", str(base_path) + "_ml.p")
        ci_path = Path(data_path, "planet_population", str(base_path) + "_ci.p")
        kde_path = Path(data_path, "planet_population", str(base_path) + "_kde.p")

        # Generate the chains and RV curve
        if rv_curve_path.exists():
            # print(rv_curve_path)
            rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
        else:
            rv_df = planet.simulate_rv_observations(rv_times, rv_error)
            rv_df.to_csv(rv_curve_path)

        if chains_path.exists():
            chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
        else:
            params = fun.gen_radvel_params(planet, param_error)
            post = fun.gen_posterior(rv_df, params, param_error)
            chains = fun.gen_chains(post)
            with open(post_path, "wb") as f:
                pickle.dump(post, f)
            chains.to_csv(chains_path)

        # Create the orbit fits
        mg_fit_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        if mg_path.exists():
            with open(mg_path, "rb") as f:
                mg = pickle.load(f)
        else:
            mg = PlanetPopulation(
                "Multivariate Gaussian",
                dist,
                Ms,
                [1],
                mg_plot_kwargs,
                mg_scatter_kwargs,
                chains=chains,
                options=mg_fit_options,
            )
            with open(mg_path, "wb") as f:
                pickle.dump(mg, f)

        mg_list.append(mg)
        # Credible interval planet population
        ci_chain = chains.quantile([0.5])
        ci_planet_inputs = {
            "period": ci_chain["per1"].values[0] * u.d,
            "secosw": ci_chain["secosw1"].values[0],
            "sesinw": ci_chain["sesinw1"].values[0],
            "K": ci_chain["k1"].values[0],
            "T_c": Time(ci_chain["tc1"].values[0], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }

        ci_planet = Planet(
            "Credible interval",
            dist,
            Ms,
            [1],
            ci_plot_kwargs,
            ci_scatter_kwargs,
            rv_inputs=ci_planet_inputs,
        )

        ci_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
        # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
        ci_std = chains.std()
        if ci_path.exists():
            with open(ci_path, "rb") as f:
                ci = pickle.load(f)
        else:
            ci = PlanetPopulation(
                "Credible interval population",
                dist,
                Ms,
                [1],
                ci_plot_kwargs,
                ci_scatter_kwargs,
                options=ci_pp_options,
                base_planet=ci_planet,
                base_planet_errors=ci_std,
            )
            with open(ci_path, "wb") as f:
                pickle.dump(ci, f)
        ci_list.append(ci)
        # Find the maximimum likelihood planet
        ml_chain = chains.loc[chains["lnprobability"].idxmax()]
        ml_planet_inputs = {
            "period": ml_chain["per1"] * u.d,
            "secosw": ml_chain["secosw1"],
            "sesinw": ml_chain["sesinw1"],
            "K": ml_chain["k1"],
            "T_c": Time(ml_chain["tc1"], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        ml_planet = Planet(
            "Max likelihood",
            dist,
            Ms,
            [1],
            ml_plot_kwargs,
            ml_scatter_kwargs,
            rv_inputs=ml_planet_inputs,
        )
        ml_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        if ml_path.exists():
            with open(ml_path, "rb") as f:
                ml = pickle.load(f)
        else:
            ml = PlanetPopulation(
                "Max likelihood",
                dist,
                Ms,
                [1],
                ml_plot_kwargs,
                ml_scatter_kwargs,
                options=ml_pp_options,
                base_planet=ml_planet,
            )
            with open(ml_path, "wb") as f:
                pickle.dump(ml, f)
        ml_list.append(ml)

        # Find the gaussian kde planet
        kde = mg.get_kde_estimate(chains)
        kde_vals = kde.x
        kde_planet_inputs = {
            "period": kde_vals[0] * u.d,
            "T_c": Time(kde_vals[1], format="jd"),
            "secosw": kde_vals[2],
            "sesinw": kde_vals[3],
            "K": kde_vals[4],
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": p,
        }
        kde_planet = Planet(
            "Max kernel density estimate",
            dist,
            Ms,
            [1],
            kde_plot_kwargs,
            kde_scatter_kwargs,
            rv_inputs=kde_planet_inputs,
        )
        kde_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": p,
        }
        if kde_path.exists():
            with open(kde_path, "rb") as f:
                kde = pickle.load(f)
        else:
            kde = PlanetPopulation(
                "Kernel density estimate",
                dist,
                Ms,
                [1],
                kde_plot_kwargs,
                kde_scatter_kwargs,
                options=kde_pp_options,
                base_planet=kde_planet,
            )
            with open(kde_path, "wb") as f:
                pickle.dump(kde, f)
        kde_list.append(kde)
        vals = ml.e
        print(f"median: {np.median(vals)}\nstd: {np.std(vals)}\n")

    pop_list = [ml_list, kde_list, ci_list, mg_list]
    rows = ["Max likelihood", "Max KDE", "Credible interval", "Multivariate Gaussian"]
    # period_nums = [0, 4, 8]
    # period_nums = [0, 1]
    # plot_times = Time([final_rv_time.decimalyear + period*T.to(u.yr).value for period in period_nums], format='decimalyear')
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, periods+1), format='decimalyear')
    # plot_times = Time(np.linspace(final_rv_time.decimalyear, final_rv_time.decimalyear + periods*T.to(u.yr).value, (periods*2+1)), format='decimalyear')
    # plot_times = Time([2010, 2020], format='decimalyear')

    # Set up paths for video
    save_location = Path("plots/construction_method_vs_error")
    time_lapse_pic_path = Path(save_location, "video/time_lapse_%04d.png")
    time_lapse_vid_path = Path(save_location, "time_lapse.mp4")

    # Remove pictures from the last run
    delete_pictures_in_path(Path(save_location, "video"))

    # Color map for the density plot
    cmap = plt.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    # set alpha for last element
    # my_cmap[:,-1] = np.linsspace(0,1,cmap.N)
    # my_cmap[:128,-1] = np.zeros(128)
    # my_cmap[:32,-1] = np.zeros(32)
    my_cmap[:32, -1] = np.zeros(32)
    my_cmap = ListedColormap(my_cmap)
    levels = np.logspace(-5, 0, 16)
    times = Time(np.linspace(2010, 2015, 100), format="decimalyear")
    for k, time_k in enumerate(tqdm(times)):
        fig, axes = plt.subplots(nrows=len(pop_list), ncols=len(rv_errors))
        construction_method_vs_error_path = Path(
            "plots", "construction_method_vs_error", "video", f"time_lapse_{k:04}.png"
        )
        cols = [
            f"Ratio: {ratio.value}"
            for ratio in rv_errors / fixed_K_val.decompose().value
        ]
        # rows = [f'Period: {period}\nYear: {year:.0f}' for period, year in zip( period_nums, plot_times.decimalyear )]
        pad = 5
        scatter = False
        xx_list = []
        yy_list = []
        f_list = []
        sep_list = []
        dMag_list = []
        cmap_scale = [None, None]
        for i, pop in enumerate(pop_list):
            # Get the true planet's location
            for j, error in enumerate(rv_errors):
                planet = planet_list[j]
                s_p, dMag_p = planet.prop_for_imaging(time_k)
                population = pop[j]
                s, dMag = population.prop_for_imaging(time_k)
                if scatter:
                    ax.scatter(
                        s.to(u.arcsec).value,
                        dMag,
                        label=f"Error of {error.decompose().value:.2f} m/s",
                        s=0.1,
                        alpha=0.2,
                    )
                else:
                    separation_for_plot = s.to(u.arcsec)
                    # xx, yy = np.mgrid[0.9*min(separation_for_plot):1.1*max(separation_for_plot):100j, 0.9*min(dMag):1.1*max(dMag):100j]
                    xx, yy = np.mgrid[0:0.45:100j, 18:34:100j]
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    values = np.vstack([separation_for_plot.value, dMag])
                    kernel = gaussian_kde(values)
                    f = np.reshape(kernel(positions).T, xx.shape)
                    f_max = np.amax(f)
                    f_min = np.amin(f)
                    if cmap_scale[0] is None or cmap_scale[0] > f_min:
                        cmap_scale[0] = f_min
                    if cmap_scale[1] is None or cmap_scale[1] < f_max:
                        cmap_scale[1] = f_max

                    xx_list.append(xx)
                    yy_list.append(yy)
                    f_list.append(f)
                    sep_list.append(s_p.to(u.arcsec))
                    dMag_list.append(dMag_p)
        num = 0
        for i, pop in enumerate(pop_list):
            for j, error in enumerate(rv_errors):
                planet = planet_list[j]
                ax = axes[i, j]
                f_scaled = f_list[num] / cmap_scale[1]
                # print(f'Scaled f: {np.amax(f_scaled)}\nUnscaled f: {np.amax(f)}')
                cfset = ax.contourf(
                    xx_list[num],
                    yy_list[num],
                    f_scaled,
                    levels=levels,
                    norm=mpl.colors.LogNorm(),
                )
                # cfset = ax.contourf(xx, yy, f_scaled, locator=mpl.ticker.LogLocator(), cmap=cmap)
                # fig.colorbar(cfset, ax=ax)
                # ax.set_title(f"Appearance after {time.decimalyear - final_rv_time.decimalyear:.1f} years")
                if ax.is_first_row():
                    ax.annotate(
                        cols[j],
                        xy=(0.5, 1),
                        xytext=(0, pad),
                        xycoords="axes fraction",
                        textcoords="offset points",
                        ha="center",
                        va="baseline",
                    )
                if ax.is_last_row():
                    ax.set_xticks(np.arange(0, 0.45, 0.1))
                    ax.set_xlabel(r"$\alpha$ (arcsec)")
                else:
                    ax.set_xticks([])

                if ax.is_first_col():
                    ax.annotate(
                        rows[i],
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - pad, 0),
                        xycoords=ax.yaxis.label,
                        textcoords="offset points",
                        ha="right",
                        va="center",
                        size="small",
                        rotation=90,
                    )
                    ax.set_yticks(np.arange(18, 34, 4))
                    ax.set_ylabel(r"$\Delta_{mag}$")
                else:
                    ax.set_yticks([])

                # Plot the actual planet's location
                ax.scatter(
                    sep_list[num],
                    dMag_list[num],
                    **planet.scatter_kwargs,
                )
                ax.set_xlim([0, 0.3])
                ax.set_ylim([18, 30])
                num += 1
        # ax.plot([IWA.to(u.arcsec).value, OWA.to(u.arcsec).value], [dMag0, dMag0], linewidth=0.5, color='black')
        # ax.plot([IWA.to(u.arcsec).value, IWA.to(u.arcsec).value], [0, dMag0], linewidth=0.5, color='black')
        # fig.legend(markerscale=20)
        # fig.colorbar(cfset)
        fig.suptitle(f"{time_k.decimalyear:.2f}")
        fig.subplots_adjust(left=0.2, top=0.9)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(cfset, cax=cbar_ax)
        cbar.set_ticks([10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e-0])
        cbar.set_label(r"Normalized Density (arcsec $^{-1}$ $\Delta_{mag}$ $^{-1}$)")
        fig.savefig(construction_method_vs_error_path, dpi=300)
    # Now combining all of the images into a video
    gen_videos(str(time_lapse_pic_path), str(time_lapse_vid_path), 25)


def delete_pictures_in_path(path):
    """
    Helper method that just deletes things
    """
    for file in os.scandir(path):
        os.unlink(file.path)


def gen_videos(input_path, output_path, framerate):
    """
    Helper method for creating videos
    """
    (ffmpeg.input(input_path, framerate=framerate).output(output_path).run())


def consistent_orbit_comparison(data_path):
    """
    Creates prob_det plot with all four construction methods, does not show the failure modes
    """
    # The actual parameters of the system
    true_a = 1.5 * u.AU
    true_e = 0
    true_w_s = 1 * u.rad  # This is the star's, to get the planet add 180 deg
    true_w_p = (true_w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
    true_W = 1 * u.rad
    true_I = np.pi / 2 * u.rad
    fixed_K_val = 1 * u.m / u.s
    # true_Mp = 1.5 * u.M_earth
    # true_Rp = 1.5 * u.R_earth
    # true_Rp = Planet.RfromM(None, true_Mp)[0]
    true_Ms = 1 * u.M_sun
    true_p = 0.367
    true_f_sed = [3]
    true_fe = [1]
    true_RV_M0 = 0 * u.rad
    breakpoint()

    T = 2 * np.pi * np.sqrt(true_a ** 3 / (true_Ms * const.G))
    true_Mp = (
        fixed_K_val
        * (true_Ms) ** (2 / 3)
        * np.sqrt(1 - true_e ** 2)
        / np.sin(true_I.to(u.rad))
        * (T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose()
    true_Rp = Planet.RfromM(None, true_Mp.to(u.M_earth).value)[0]

    # This is the mean anomaly at the time of the first RV measurment
    dist = 10 * u.pc

    # This is the number of samples to take for the planet populations
    n_fits = 50000

    # Create figures
    rv_errors = [1, 0.1] * u.m / u.s  # This is in m/s
    fig, axes = plt.subplots(figsize=(10, 4 * len(rv_errors)), nrows=2)  # was 10,4

    #######
    # Times of RV observations
    #######
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)

    # This is used to tune how close to the real value the guesses are
    param_error = 0.1

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    ########
    # Plot kwargs
    ########
    cmap = mpl.cm.get_cmap("viridis")
    # cmap = mpl.cm.get_cmap('nipy_spectral')
    colors = cmap(np.linspace(0, 0.95, 5))
    # True planet kwargs
    true_scatter_kwargs = {
        "color": "lime",
        "alpha": 1,
        "s": 250,
        "marker": "P",
        "label": "True planet",
        "edgecolors": "black",
        "linewidth": 2,
    }
    true_plot_kwargs = {
        "color": colors[0],
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "--",
        # "linestyle": "dashdot",
    }

    # kwargs used for the actual RV fits
    rv_scatter_kwargs = {
        "color": "white",
        "alpha": 0.1,
        "s": 0.01,
        "label": "Multivariate Gaussian",
    }
    rv_plot_kwargs = {
        "color": colors[4],
        "alpha": 1,
        "label": "Multivariate Gaussian",
        "linewidth": 1,
        # "linestyle": "-",
    }

    # Credible interval kwargs
    ci_scatter_kwargs = {
        "color": "blue",
        "alpha": 1,
        "s": 10,
        "label": "Credible interval",
        "marker": ">",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    ci_plot_kwargs = {
        "color": colors[3],
        "alpha": 1,
        "label": "Credible interval",
        "markersize": 5,
        # "linestyle": "dashed",
        "linewidth": 1,
    }

    # Maximum likelihood kwargs
    ml_scatter_kwargs = {
        "color": "brown",
        "alpha": 1,
        "s": 10,
        "label": "Max likelihood",
        "marker": "x",
        "edgecolors": "white",
    }
    ml_plot_kwargs = {
        "color": colors[2],
        "alpha": 1,
        "label": "Max likelihood",
        "markersize": 5,
        # "linestyle": "dotted",
        "linewidth": 1,
    }

    # Kernel density estimate kwargs
    kde_scatter_kwargs = {
        "color": "green",
        "alpha": 1,
        "s": 10,
        "label": "Max KDE",
        "marker": "*",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    kde_plot_kwargs = {
        "color": colors[1],
        "alpha": 1,
        "label": "Max KDE",
        "markersize": 5,
        "linewidth": 1,
        # "linestyle": (0, (5,1)),# "linestyle": (0, (5, 10)),
    }

    for k, ax in enumerate(axes):
        rv_error = rv_errors[k]
        ratio = rv_error / fixed_K_val
        ###########
        # Create the true_planet
        ##########
        # The inputs for the planet class
        true_inputs = {
            "a": true_a,
            "e": true_e,
            "W": true_W,
            "I": true_I,
            "w": true_w_p,
            "Mp": true_Mp,
            "Rp": true_Rp,
            "f_sed": true_f_sed,
            "p": true_p,
            "M0": true_RV_M0,
            "t0": initial_rv_time,
            "rv_error": rv_error,
        }

        # Cram all the information into the class
        true_planet = Planet(
            "True planet",
            dist,
            true_Ms,
            [1],
            true_plot_kwargs,
            true_scatter_kwargs,
            keplerian_inputs=true_inputs,
        )

        #######
        # Create the path locations
        #######
        base_path = true_planet.gen_base_path(
            rv_error, initial_rv_time, final_rv_time, IWA, OWA, dMag0
        )
        rv_curve_path = Path(data_path, "rv_curve", base_path).with_suffix(".csv")
        post_path = Path(data_path, "post", base_path).with_suffix(".p")
        chains_path = Path(data_path, "chains", base_path).with_suffix(".csv")

        if rv_curve_path.exists():
            rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
        else:
            rv_df = true_planet.simulate_rv_observations(rv_times, rv_error)
            # rv_df = RV_data_gen(true_planet, rv_times, rv_error)
            rv_df.to_csv(rv_curve_path)

        if chains_path.exists():
            chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
            with open(post_path, "rb") as open_path:
                post = pickle.load(open_path)
        else:
            params = fun.gen_radvel_params(true_planet, param_error)
            post = fun.gen_posterior(rv_df, params, param_error)
            chains = fun.gen_chains(post)
            with open(post_path, "wb") as save_path:
                pickle.dump(post, save_path)
            chains.to_csv(chains_path)
        ###############
        # Create the population of planets using the chains
        ###############
        # rv_fit is the multivariate gaussian
        rv_fit_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": 3,
            "fixed_p": true_p,
        }
        rv_fits = PlanetPopulation(
            "RV fits",
            dist,
            true_Ms,
            true_fe,
            rv_plot_kwargs,
            rv_scatter_kwargs,
            chains=chains,
            options=rv_fit_options,
        )
        # if not covariance_path.exists():
        # with open(covariance_path, 'wb') as save_path:
        # pickle.dump(rv_fits.cov_df, save_path)
        # Create the credible interval planet
        ci_chain = chains.quantile([0.5])
        ci_planet_inputs = {
            "period": ci_chain["per1"].values[0] * u.d,
            "secosw": ci_chain["secosw1"].values[0],
            "sesinw": ci_chain["sesinw1"].values[0],
            "K": ci_chain["k1"].values[0],
            "T_c": Time(ci_chain["tc1"].values[0], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": true_p,
        }

        ci_planet = Planet(
            "Credible interval",
            dist,
            true_Ms,
            true_fe,
            ci_plot_kwargs,
            ci_scatter_kwargs,
            rv_inputs=ci_planet_inputs,
        )

        ci_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": true_p,
        }
        # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
        # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
        ci_std = chains.std()
        ci_planet_population = PlanetPopulation(
            "Credible interval population",
            dist,
            true_Ms,
            true_fe,
            ci_plot_kwargs,
            ci_scatter_kwargs,
            options=ci_pp_options,
            base_planet=ci_planet,
            base_planet_errors=ci_std,
        )

        # Find the maximimum likelihood planet
        ml_chain = chains.loc[chains["lnprobability"].idxmax()]
        ml_planet_inputs = {
            "period": ml_chain["per1"] * u.d,
            "secosw": ml_chain["secosw1"],
            "sesinw": ml_chain["sesinw1"],
            "K": ml_chain["k1"],
            "T_c": Time(ml_chain["tc1"], format="jd"),
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": true_p,
        }
        ml_planet = Planet(
            "Max likelihood",
            dist,
            true_Ms,
            true_fe,
            ml_plot_kwargs,
            ml_scatter_kwargs,
            rv_inputs=ml_planet_inputs,
        )
        ml_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": true_p,
        }
        # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
        # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
        ml_planet_population = PlanetPopulation(
            "Maximum likelihood population",
            dist,
            true_Ms,
            true_fe,
            ml_plot_kwargs,
            ml_scatter_kwargs,
            options=ml_pp_options,
            base_planet=ml_planet,
        )

        # Find the gaussian kde planet
        kde = rv_fits.get_kde_estimate(chains)
        kde_vals = kde.x
        kde_planet_inputs = {
            "period": kde_vals[0] * u.d,
            "T_c": Time(kde_vals[1], format="jd"),
            "secosw": kde_vals[2],
            "sesinw": kde_vals[3],
            "K": kde_vals[4],
            "fixed_inc": np.pi * u.rad / 2,
            "fixed_f_sed": 3,
            "fixed_p": true_p,
        }
        kde_planet = Planet(
            "Max kernel density estimate",
            dist,
            true_Ms,
            true_fe,
            kde_plot_kwargs,
            kde_scatter_kwargs,
            rv_inputs=kde_planet_inputs,
        )
        kde_pp_options = {
            "n_fits": n_fits,
            "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
            "cov_samples": 1000,
            "fixed_inc": None,
            "fixed_f_sed": None,
            "fixed_p": true_p,
        }
        # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
        # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
        kde_planet_population = PlanetPopulation(
            "Max kernel density estimate population",
            dist,
            true_Ms,
            true_fe,
            kde_plot_kwargs,
            kde_scatter_kwargs,
            options=kde_pp_options,
            base_planet=kde_planet,
        )
        # Create the plot showing the different probability of detections
        initial_comp_time = Time(2010.0, format="decimalyear")
        final_comp_time = Time(2030, format="decimalyear")
        comp_observations = 1000
        comp_times = Time(
            np.linspace(initial_comp_time.jd, final_comp_time.jd, comp_observations),
            format="jd",
        )
        # fun.point_cloud_comparison(
        # [true_planet, kde_planet_population, ml_planet_population, ci_planet_population, rv_fits],
        # # [true_planet, rv_fits, ci_planet_population, ml_planet_population, kde_planet_population],
        # comp_times,
        # dMag0,
        # IWA,
        # OWA,
        # rv_error,
        # )

        # Moving the point_cloud_comparison function into this file
        planet_list = [
            true_planet,
            kde_planet_population,
            ml_planet_population,
            ci_planet_population,
            rv_fits,
        ]
        times = comp_times
        times_progressed = []
        # Calculate the working angle contrasts

        # plt.style.use("dark_background")
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 15
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        # plt.rcParams['legend.fontsize']=12
        for t in tqdm(times, desc="Probability of detection"):
            # Separation flux and completeness plots
            times_progressed.append(Time(t, format="jd").decimalyear)
            # Defining a colormap for the cloud
            for current_planet in planet_list:
                if not hasattr(current_planet, "percent_detectable"):
                    current_planet.percent_detectable = []
                WA, dMag = current_planet.prop_for_imaging(t)

                # find the different visibility measures
                visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)

                # Compute percents
                current_planet.percent_detectable.append(
                    sum(visible) / current_planet.num
                )

        # Now make the plot
        # fig, comp_ax = plt.subplots(figsize=(12, 5)) # was 10,4
        # Set up the plots so that the legend is outside
        comp_box = ax.get_position()
        ax.set_position([comp_box.x0, comp_box.y0, comp_box.width, comp_box.height])
        ax.set_xlim(times.decimalyear[0], times.decimalyear[-1])
        ax.set_ylim([-0.10, 1.1])
        ax.set_yticks(np.linspace(0, 1, num=6))
        ax.set_ylabel("Probability of detection")
        ax.set_xlabel("Year")
        for current_planet in planet_list:
            # Bottom subplot with the completness
            ax.plot(
                times_progressed,
                current_planet.percent_detectable,
                **current_planet.plot_kwargs,
            )
        # comp_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.legend(loc="lower right", framealpha=1)
        ax.set_title(f"Ratio: {ratio:.1f}")
    fig.tight_layout()
    fname = "plots/prob_det/prob_det.pdf"
    fig.savefig(fname)
    plt.close()


def chain_creation(
    rv_times,
    initial_rv_time,
    final_rv_time,
    param_error,
    constant_elements,
    data_path,
    elements,
):
    """
    This function gets called to create all of the radvel fits, it returns the
    planet object it creates but caches the chains to save memory
    """
    a, i, e, Mp, rv_error, Rp = elements
    w_p, W, Ms, p, RV_M0, dist, K_val, IWA, OWA, dMag0 = constant_elements
    if Mp is None:
        T = 2 * np.pi * np.sqrt(a ** 3 / (Ms * const.G))
        Mp = (
            K_val
            * (Ms) ** (2 / 3)
            * np.sqrt(1 - e ** 2)
            / np.sin(i.to(u.rad))
            * (T / (2 * np.pi * const.G)) ** (1 / 3)
        ).decompose()
    ###########
    # Create the true_planet
    ##########
    scatter_kwargs = {
        "color": "lime",
        "alpha": 1,
        "s": 250,
        "marker": "P",
        "label": "True planet",
        "edgecolors": "black",
        "linewidth": 2,
    }
    plot_kwargs = {
        "color": "lime",
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "dashdot",
    }
    # The inputs for the planet class
    if Rp is None:
        Rp = Planet.RfromM(None, Mp.to(u.M_earth).value)[0]
    true_inputs = {
        "a": a,
        "e": e,
        "W": W,
        "I": i,
        "w": w_p,
        "Mp": Mp,
        "Rp": Rp,
        "f_sed": [3],
        "p": p,
        "M0": RV_M0,
        "t0": initial_rv_time,
        "rv_error": rv_error,
    }
    planet = Planet(
        "Planet",
        dist,
        Ms,
        [1],
        plot_kwargs,
        scatter_kwargs,
        keplerian_inputs=true_inputs,
    )

    base_path = planet.gen_base_path(
        rv_error, initial_rv_time, final_rv_time, IWA, OWA, dMag0
    )
    rv_curve_path = Path(data_path, "rv_curve", str(base_path) + ".csv")
    post_path = Path(data_path, "post", str(base_path) + ".p")
    chains_path = Path(data_path, "chains", str(base_path) + ".csv")

    # Generate the chains and RV curve
    if rv_curve_path.exists():
        rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
    else:
        rv_df = planet.simulate_rv_observations(rv_times, rv_error)
        rv_df.to_csv(rv_curve_path)

    if chains_path.exists():
        pass
    else:
        params = fun.gen_radvel_params(planet, param_error)
        post = fun.gen_posterior(rv_df, params, param_error)
        chains = fun.gen_chains(post)
        with open(post_path, "wb") as save_path:
            pickle.dump(post, save_path)
        chains.to_csv(chains_path)
    return planet


def create_populations(
    dist,
    Ms,
    fit_options,
    plot_kwargs,
    scatter_kwargs,
    times,
    dMag0,
    IWA,
    OWA,
    data_path,
    pop_types,
    planet,
):
    """
    This function will create the planet population and run the time until failure calculations,
    all it needs for inputs are the necessary parts to create a planet_population
    """
    # Get the location of the chains so we aren't holding all in memory
    t0 = time.time()
    chains_path = Path(data_path, "chains", str(planet.base_path) + ".csv")
    chains_loaded = False

    # problem_child = 'a2.00AU_e0.00_W0.00rad_I2.31rad_w0.00rad_Mp21.49earthMass_Rp4.03earthRad_Ms1.00solMass_error0.1000_ti2000.00_tf2010.00_IWA0.058arcsec_OWA6.0arcsec_dMag026.5_imgtimei2010_imgtimef2030_imgtimestep1.0d.p'
    # problem_child = 'a2.00AU_e0.00_W0.00rad_I2.31rad_w0.00rad_Mp21.49earthMass_Rp4.03earthRad_Ms1.00solMass_error0.1000'

    if "MG" in pop_types:
        MG_fits_path = Path(
            data_path,
            "planet_population",
            str(planet.base_path)
            + (
                f"_imgtimei{times[0].decimalyear:.0f}_imgtimef{times[-1].decimalyear:.0f}_imgtimestep{(times[1].decimalyear*u.yr-times[0].decimalyear*u.yr).to(u.d):.1f}_MG".replace(
                    " ", ""
                )
                + ".p"
            ),
        )
        if not MG_fits_path.exists():
            # kwargs used for the actual RV fits
            chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
            chains_loaded = True
            MG_fits = PlanetPopulation(
                "Multivariate Gaussian",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                chains=chains,
                options=fit_options,
            )
            # Find the first time that the planet is above the threshold of failure longer than the failure_duration
            MG_fits.percent_detectable = []
            for t in tqdm(times):
                WA, dMag = MG_fits.prop_for_imaging(t)
                visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
                MG_fits.percent_detectable.append(sum(visible) / MG_fits.num)
            with open(MG_fits_path, "wb") as save_path:
                pickle.dump(MG_fits, save_path)
    if "MKDE" in pop_types:
        KDE_fits_path = Path(
            data_path,
            "planet_population",
            str(planet.base_path)
            + (
                f"_imgtimei{times[0].decimalyear:.0f}_imgtimef{times[-1].decimalyear:.0f}_imgtimestep{(times[1].decimalyear*u.yr-times[0].decimalyear*u.yr).to(u.d):.1f}_MKDE".replace(
                    " ", ""
                )
                + ".p"
            ),
        )
        try:
            # If MG_fits was created on a previous run that this doesn't work, so here's a hacky way around that with a try-except
            MG_fits
        except:
            MG_fits_path = Path(
                data_path,
                "planet_population",
                str(planet.base_path)
                + (
                    f"_imgtimei{times[0].decimalyear:.0f}_imgtimef{times[-1].decimalyear:.0f}_imgtimestep{(times[1].decimalyear*u.yr-times[0].decimalyear*u.yr).to(u.d):.1f}_MG".replace(
                        " ", ""
                    )
                    + ".p"
                ),
            )
            with open(MG_fits_path, "rb") as file:
                MG_fits = pickle.load(file)
        if not KDE_fits_path.exists():
            if not chains_loaded:
                chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
                chains_loaded = True
            KDE = MG_fits.get_kde_estimate(chains)
            KDE_vals = KDE.x
            KDE_planet_inputs = {
                "period": KDE_vals[0] * u.d,
                "T_c": Time(KDE_vals[1], format="jd"),
                "secosw": KDE_vals[2],
                "sesinw": KDE_vals[3],
                "K": KDE_vals[4],
                "fixed_inc": np.pi * u.rad / 2,
                "fixed_f_sed": 3,
                "fixed_p": 0.367,
            }
            KDE_planet = Planet(
                "Max kernel density estimate",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                rv_inputs=KDE_planet_inputs,
            )
            KDE_fits = PlanetPopulation(
                "Max kernel density estimate population",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                options=fit_options,
                base_planet=KDE_planet,
            )
            # Find the first time that the planet is above the threshold of failure longer than the failure_duration
            KDE_fits.percent_detectable = []
            for t in tqdm(times):
                WA, dMag = KDE_fits.prop_for_imaging(t)
                visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
                KDE_fits.percent_detectable.append(sum(visible) / KDE_fits.num)
            with open(KDE_fits_path, "wb") as save_path:
                pickle.dump(KDE_fits, save_path)

    if "CI" in pop_types:
        CI_fits_path = Path(
            data_path,
            "planet_population",
            str(planet.base_path)
            + (
                f"_imgtimei{times[0].decimalyear:.0f}_imgtimef{times[-1].decimalyear:.0f}_imgtimestep{(times[1].decimalyear*u.yr-times[0].decimalyear*u.yr).to(u.d):.1f}_CI".replace(
                    " ", ""
                )
                + ".p"
            ),
        )
        if not CI_fits_path.exists():
            # Find the first time that the planet is above the threshold of failure longer than the failure_duration
            if not chains_loaded:
                chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
                chains_loaded = True
            CI_chain = chains.quantile([0.5])
            CI_planet_inputs = {
                "period": CI_chain["per1"].values[0] * u.d,
                "secosw": CI_chain["secosw1"].values[0],
                "sesinw": CI_chain["sesinw1"].values[0],
                "K": CI_chain["k1"].values[0],
                "T_c": Time(CI_chain["tc1"].values[0], format="jd"),
                "fixed_inc": np.pi * u.rad / 2,
                "fixed_f_sed": 3,
                "fixed_p": 0.367,
            }

            CI_planet = Planet(
                "Credible interval",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                rv_inputs=CI_planet_inputs,
            )

            CI_pp_options = {
                "n_fits": fit_options['n_fits'],
                "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
                "cov_samples": 1000,
                "fixed_inc": None,
                "fixed_f_sed": None,
                "fixed_p": 0.367,
            }
            # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
            # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
            CI_std = chains.std()
            CI_fits = PlanetPopulation(
                "Credible interval population",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                options=CI_pp_options,
                base_planet=CI_planet,
                base_planet_errors=CI_std,
            )
            CI_fits.percent_detectable = []
            for t in tqdm(times):
                WA, dMag = CI_fits.prop_for_imaging(t)
                visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
                CI_fits.percent_detectable.append(sum(visible) / CI_fits.num)
            with open(CI_fits_path, "wb") as save_path:
                pickle.dump(CI_fits, save_path)

    if "ML" in pop_types:
        ML_fits_path = Path(
            data_path,
            "planet_population",
            str(planet.base_path)
            + (
                f"_imgtimei{times[0].decimalyear:.0f}_imgtimef{times[-1].decimalyear:.0f}_imgtimestep{(times[1].decimalyear*u.yr-times[0].decimalyear*u.yr).to(u.d):.1f}_ML".replace(
                    " ", ""
                )
                + ".p"
            ),
        )
        if not ML_fits_path.exists():
            # Find the maximimum likelihood planet
            if not chains_loaded:
                chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
                chains_loaded = True
            ML_chain = chains.loc[chains["lnprobability"].idxmax()]
            ML_planet_inputs = {
                "period": ML_chain["per1"] * u.d,
                "secosw": ML_chain["secosw1"],
                "sesinw": ML_chain["sesinw1"],
                "K": ML_chain["k1"],
                "T_c": Time(ML_chain["tc1"], format="jd"),
                "fixed_inc": np.pi * u.rad / 2,
                "fixed_f_sed": 3,
                "fixed_p": 0.367,
            }
            ML_planet = Planet(
                "Max likelihood",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                rv_inputs=ML_planet_inputs,
            )
            # ci_chain_low = ci_chain - rv_fits.chains.quantile(0.159)
            # ci_chain_high = rv_fits.chains.quantile(0.841) - ci_chain
            ML_fits = PlanetPopulation(
                "Maximum likelihood population",
                dist,
                Ms,
                [1],
                plot_kwargs,
                scatter_kwargs,
                options=fit_options,
                base_planet=ML_planet,
            )
            ML_fits.percent_detectable = []
            for t in tqdm(times):
                WA, dMag = ML_fits.prop_for_imaging(t)
                visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
                ML_fits.percent_detectable.append(sum(visible) / ML_fits.num)
            with open(ML_fits_path, "wb") as save_path:
                pickle.dump(ML_fits, save_path)
    # print(f"Time elapsed for population creation: {(time.time()-t0)/60:.2f} mins\n")
    # print(f'Dumped {rv_fits_path}\r')
    # return rv_fits


def long_MG_bar_plot(data_path, cores, inclinations, a_setting):
    """
    This is used for the long run with more inclinations sampled for the multivariate Gaussian construction method
    """
    # So I want to have it set up for different settings easily using true/false inputs
    fixed_K = True  # Samples I and then matches Mp
    fixed_K_val = 1 * u.m / u.s
    i_range = [90, 170] * u.deg
    i_path = Path(data_path, f"i_vals{i_range[0].value}-{i_range[1].value}.p")
    i_n = (
        4 * inclinations
    )  # This is used to determine how many inclination values to use

    fixed_a_val = 2 * u.AU

    # rv_error_range = [0.1, 1.25, 2] * u.m / u.s  # This is in m/s
    rv_error_range = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 2] * u.m / u.s  # This is in m/s

    # Ranges for the values
    e_range = [0]

    if a_setting:
        a_range = [fixed_a_val.to(u.AU)] * u.AU
    else:
        a_range = np.linspace(1, 2, num=10) * u.AU
    if fixed_K:
        if i_path.exists():
            with open(i_path, "rb") as file_name:
                truncated_curve = pickle.load(file_name)
            # truncated_curve = pickle.load(open(i_path, 'rb'))
        else:
            sin_curve = np.arccos(1 - 2 * np.random.uniform(size=1000000)) * u.rad
            truncated_curve = sin_curve[
                (sin_curve > i_range[0]) & (sin_curve < i_range[1])
            ]
            with open(i_path, "wb") as file_name:
                pickle.dump(truncated_curve, file_name)
            # pickle.dump(truncated_curve, open(i_path, 'wb'))
        Mp_range = [None]
        # i_range = truncated_curve[:i_n]
        i_range = truncated_curve[:i_n]
    else:
        # The ranges for the different simulations
        i_range = np.linspace(90, 170, num=5) * u.deg
        Mp_range = np.array([1, 2, 20, 100, 300]) * u.M_earth

    # Constant values for all the planets
    w_p = 0 * u.rad
    W = 0 * u.rad
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    base_elements = (w_p, W, Ms, p, RV_M0, dist, fixed_K_val, IWA, OWA, dMag0)

    # Number of consistent orbits to generate
    n_fits = 50000

    # RV observation times
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)
    param_error = 0.1  # Also m/s

    ##########
    # Direct imaging observation times
    #########
    initial_obs_time = Time(2010.0, format="decimalyear")
    final_obs_time = Time(2030, format="decimalyear")
    timestep = 1 * u.d
    # timestep = 0.5*u.week
    # obs_n = 10
    # img_times = Time(np.linspace(initial_obs_time.jd, final_obs_time.jd, obs_n), format='jd')
    img_times = Time(
        np.arange(initial_obs_time.jd, final_obs_time.jd, step=timestep.to("d").value),
        format="jd",
    )

    #### Roman Space Telescope performance
    # IWA = 0.19116374 * u.arcsec
    # OWA = 0.57349123 * u.arcsec
    # dMag0 = 20.5

    print("Creating/checking chains")
    partial_chain_creation = functools.partial(
        chain_creation,
        rv_times,
        initial_rv_time,
        final_rv_time,
        param_error,
        base_elements,
        data_path,
    )
    element_list = list(product(a_range, i_range, e_range, Mp_range, rv_error_range))
    with ProcessPoolExecutor(max_workers=int(cores / 2.5)) as executor:
        # with ProcessPoolExecutor(max_workers=20) as executor: #tphon
        results = executor.map(partial_chain_creation, element_list)

    planet_list = list(results)
    # Now that the chains have all been generated we can do the propagation

    rv_fit_options = {
        "n_fits": n_fits,
        "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
        "cov_samples": 1000,
        "fixed_inc": None,
        "fixed_f_sed": 3,
        "fixed_p": p,
    }

    rv_scatter_kwargs = {
        "color": "white",
        "alpha": 0.1,
        "s": 0.01,
        "label": "RV fit cloud",
    }
    rv_plot_kwargs = {
        "color": "white",
        "alpha": 1,
        "label": "RV fit cloud",
        "linewidth": 1,
        "linestyle": "-",
    }
    print("Creating populations")
    pop_types = ["MG"]
    partial_population_creation = functools.partial(
        create_populations,
        dist,
        Ms,
        rv_fit_options,
        rv_plot_kwargs,
        rv_scatter_kwargs,
        img_times,
        dMag0,
        IWA,
        OWA,
        data_path,
        pop_types,
    )
    with ProcessPoolExecutor(max_workers=cores) as executor:
        executor.map(partial_population_creation, planet_list)
    # with ProcessPoolExecutor(max_workers=48) as executor: #tphon
    # for planet in planet_list:
    # # Debugging version
    # population = create_populations(dist, Ms, rv_fit_options, rv_plot_kwargs, rv_scatter_kwargs, img_times, dMag0, IWA, OWA,data_path, ['MG'], planet)

    # Calculate all the ratios
    ratios = []
    for planet in planet_list:
        ratio = planet.rv_error.decompose().value / planet.K.decompose().value
        ratios.append(round(ratio, 2))
    unique_ratios = np.unique(ratios)

    # Delete the folders
    types_of_plots = ["failure_plots"]
    # types_of_failure = ['no_failure', 'amplitude_failure', 'period_failure', 'intermittent_failure']
    types_of_failure = [
        "no_failure_(not_always_detectable)",
        "no_failure_(always_detectable)",
        "intermittent_failure",
        "dispersion_failure",
        "both_failure",
    ]
    fail_path = Path("plots", "failure_plots")
    if not fail_path.exists():
        os.mkdir(fail_path)
    for ratio in unique_ratios:
        directory = Path("plots/failure_plots/", f"{ratio:.2f}")
        directory = Path("plots/failure_plots/")
        for folder in directory.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)

    # Recreate the folders necessary
    for plot_type in types_of_plots:
        plot_path = Path("plots", plot_type)
        if not plot_path.exists():
            os.mkdir(plot_path)
        for ratio in unique_ratios:
            ratio_path = Path(plot_path, f"{ratio:.2f}")
            if not ratio_path.exists():
                os.mkdir(ratio_path)
            for failure_type in types_of_failure:
                failure_path = Path(ratio_path, failure_type)
                if not failure_path.exists():
                    os.mkdir(failure_path)

    # Keep track of failures
    failure_modes = []
    list_of_ratios = []

    # Creating arrays to keep track of the times of first intermittent
    # failure and dispersion failure
    first_if_list = []

    # This keeps track of the rate of intermittent failure for all that fail
    failure_rates = []

    # Defining colorscheme
    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.95, 5))

    # Calculate the failure modes
    for i, planet in enumerate(tqdm(planet_list)):
        failure_mode, failure_rates, first_if_list = failure_mode_calculations(
            data_path,
            planet,
            img_times,
            [IWA, OWA, dMag0],
            failure_rates,
            first_if_list,
            create_prob_det_plots=False,
            colors=colors,
        )

        failure_modes.append(failure_mode)
        list_of_ratios.append(
            round(planet.rv_error.decompose().value / planet.K.decompose().value, 2)
        )

    # Figure out the ratio breakdown of the failures
    u_ratios = np.unique(ratios)
    fail_df = pd.DataFrame(
        np.zeros((len(u_ratios), len(types_of_failure))),
        columns=types_of_failure,
        index=u_ratios,
    )
    for ratio in u_ratios:
        indices = np.where(list_of_ratios == ratio)[0]
        failures = np.array(failure_modes)[indices]
        for fail in types_of_failure:
            fail_df.at[ratio, fail] = len(np.where(failures == fail)[0])

    print(fail_df)
    fail_df.to_pickle(Path(data_path, "long_MG_fail_df.p"))
    # Create the tick and ticklabels so that the bars are evenly spaced
    x_ticks = np.linspace(0, 2, len(u_ratios))
    x_labels = [str(round(ratio, 2)) for ratio in u_ratios]
    # Create and save a bar plot with the different failure modes
    bar_path = Path("plots/failure_plots/total_bar_plot.pdf")
    fig_bar, ax_bar = plt.subplots(figsize=[7.692, 5])
    percent_fail_df = 100 * fail_df / fail_df.sum(axis=1).values[0]
    types_of_failure_labels = [
        failure[0].upper() + failure[1:].replace("_", " ")
        for failure in types_of_failure
    ]
    for i, failure_type in enumerate(types_of_failure):
        failure_label = types_of_failure_labels[i]
        if i == 0:
            ax_bar.bar(
                x_ticks,
                percent_fail_df[failure_type].to_numpy(),
                0.17,
                label=failure_label,
                color=colors[-(i + 1)],
            )
            bottom_vals = percent_fail_df[failure_type].to_numpy()
        else:
            ax_bar.bar(
                x_ticks,
                percent_fail_df[failure_type].to_numpy(),
                0.17,
                bottom=bottom_vals,
                label=failure_label,
                color=colors[-(i + 1)],
            )
            bottom_vals = bottom_vals + percent_fail_df[types_of_failure[i]].to_numpy()

    # ax = fail_df.plot.bar(stacked=True)
    ax_bar.set_xlabel("Ratio of error/semi-amplitude")
    ax_bar.set_xticks(x_ticks)
    ax_bar.set_xticklabels(x_labels)
    # ax_bar.set_xticks(np.append(np.arange(0.25, 2.1, 0.25), 0.1))
    ax_bar.set_ylabel("Percent")
    ax_bar.set_title(
        f"Multivariate Gaussian, planets per ratio: {len(planet_list)/len(u_ratios):.0f}"
    )
    handles, labels = ax_bar.get_legend_handles_labels()
    # ax_bar.legend(handles[::-1], labels[::-1], loc='lower left', framealpha=1)
    # fig = ax.get_figure()
    fig_bar.legend(handles[::-1], labels[::-1], framealpha=1, loc=7)
    fig_bar.subplots_adjust(right=0.65)
    fig_bar.savefig(bar_path, dpi=400)

    # Plot showing the times until intermittent failure for different ratios
    fif_df = pd.DataFrame.from_records(first_if_list, columns=["ratio", "fif"])
    fig_violin, (ax_fif, ax_fr) = plt.subplots(figsize=[8, 5], nrows=2)
    fif_list = []
    for ratio in u_ratios:
        # ratio_df = fif_df.loc[fif_df.ratio == ratio]
        # ax_fif.scatter(ratio_df.ratio, ratio_df.fif)
        ratio_fif_arr = fif_df.loc[fif_df.ratio == ratio].fif.to_numpy()
        fif_list.append(ratio_fif_arr)
        # ax_fif.scatter(ratio_df.ratio, ratio_df.fif)
    # ax_fif.violinplot(fif_list, u_ratios, widths = 0.1, showmedians=True)
    fif_ylims = [0, 20]
    ax_fif = violin_plot_style(ax_fif, fif_list, colors, u_ratios, True, fif_ylims)
    # ax_fif.set_xticks(np.append(np.linspace(0.25, 2, 8), 0.1))
    ax_fif.set_ylabel("Time of first intermittent failure (yr)", fontsize=8)
    ax_fif.set_yticks(np.arange(0, 21, 5))

    # Plot showing the time until dispersion failure for different ratios
    # ftf_df = pd.DataFrame.from_records(first_df_list, columns=['ratio', 'ftf'])
    # fig_ftf, ax_ftf = plt.subplots()
    # ftf_list = []
    # ftf_ratio_list = []
    # for ratio in u_ratios:
    # # ratio_df = ftf_df.loc[ftf_df.ratio == ratio]
    # # ax_ftf.scatter(ratio_df.ratio, ratio_df.ftf)
    # ratio_ftf_arr = ftf_df.loc[ftf_df.ratio == ratio].ftf.to_numpy()
    # if len(ratio_ftf_arr) == 0:
    # continue
    # else:
    # ftf_list.append(ratio_ftf_arr)
    # ftf_ratio_list.append(ratio)
    # if len(ftf_ratio_list) > 0:
    # ax_ftf.violinplot(ftf_list, ftf_ratio_list, widths = 0.1, showmedians=True)
    # # ax_ftf.violinplot(ftf_list, np.arange(0, len(ftf_ratio_list)), widths = 0.1, showmedians=True)
    # # ax_ftf.set_xticks(ftf_ratio_list)
    # ax_ftf.set_xlabel('Ratio of Error/Semi-amplitude')
    # ax_ftf.set_ylabel('Years until first dispersion failure')
    # fig_ftf.savefig('plots/failure_plots/time_till_dispersion_failure.png', dpi=300)

    # Plot of the failure rates
    failure_rates_df = pd.DataFrame.from_records(
        failure_rates, columns=["ratio", "failure_rate"]
    )
    # fig_fr, ax_fr = plt.subplots()

    fr_list = []
    for ratio in u_ratios:
        # ratio_df = failure_rates_df.loc[failure_rates_df.ratio == ratio]
        # ax_fr.scatter(ratio_df.ratio, ratio_df.failure_rate)
        ratio_arr = (
            failure_rates_df.loc[
                failure_rates_df.ratio == ratio
            ].failure_rate.to_numpy()
            * 100
        )
        fr_list.append(ratio_arr)

    fr_ylims = [0, 100]
    ax_fr = violin_plot_style(ax_fr, fr_list, colors, u_ratios, False, fr_ylims)
    ax_fr.set_xlabel("Ratio of error/semi-amplitude", fontsize=10)
    ax_fr.set_ylabel("Intermittent failure rate (%)", fontsize=8)
    # ax_fr.set_xticks(np.append(np.linspace(0.25, 2, 8), 0.1))
    # ax_fr.set_yticks(np.arange(0, 100.1, 20))
    fig_violin.tight_layout()
    fig_violin.align_ylabels()
    fig_violin.savefig("plots/failure_plots/violin_plots.pdf", dpi=300)
    plt.close("all")

    # Plot the different inclination distributions
    # int_fail_inclinations_df = pd.DataFrame.from_records(intermittent_failure_inclinations, columns=['ratio', 'inclination'])
    # dis_fail_inclinations_df = pd.DataFrame.from_records(dispersion_failure_inclinations, columns=['ratio', 'inclination'])
    # for ratio in u_ratios:
    # fig_int_fail_inc, ax_int_fail_inc = plt.subplots()
    # failure_arr = failure_rates_df.loc[failure_rates_df.ratio == ratio].failure_rate.to_numpy()
    # inc_arr = int_fail_inclinations_df.loc[int_fail_inclinations_df.ratio == ratio].inclination.to_numpy()
    # inc_deg = [x.to('deg').value for x in inc_arr]
    # ax_int_fail_inc.scatter(inc_deg, failure_arr)
    # ax_int_fail_inc.set_xlabel('Inclination (degrees)')
    # ax_int_fail_inc.set_ylabel('Intermittent Failure Rate (degrees)')
    # ax_int_fail_inc.set_title(f'Ratio {ratio}')
    # ratio_path = Path(f'plots/failure_plots/{ratio:.2f}_inclination.png')
    # fig_int_fail_inc.savefig(ratio_path, dpi=300)
    plt.close("all")


def short_all_construction_methods_bar_plot(data_path, cores, inclinations, a_setting):
    """
    This function creates the plot comparing all four construction method failure results
    """
    fixed_K = True  # Samples I and then matches Mp
    fixed_K_val = 1 * u.m / u.s
    i_range = [90, 170] * u.deg
    i_path = Path(data_path, f"i_vals{i_range[0].value}-{i_range[1].value}.p")
    i_n = inclinations  # This is used to determine how many inclination values to use

    fixed_a_val = 2 * u.AU

    # rv_error_range = [0.1, 0.25, 0.5, 0.75]*u.m/u.s  # This is in m/s
    # rv_error_range = [1, 1.25, 2]*u.m/u.s  # This is in m/s
    rv_error_range = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 2] * u.m / u.s  # This is in m/s

    # Ranges for the values
    e_range = [0]

    if a_setting:
        a_range = [fixed_a_val.to(u.AU)] * u.AU
    else:
        a_range = np.linspace(1, 2, num=10) * u.AU
    if fixed_K:
        if i_path.exists():
            with open(i_path, "rb") as f:
                truncated_curve = pickle.load(f)
        else:
            sin_curve = np.arccos(1 - 2 * np.random.uniform(size=1000000)) * u.rad
            truncated_curve = sin_curve[
                (sin_curve > i_range[0]) & (sin_curve < i_range[1])
            ]
            with open(i_path, "wb") as f:
                pickle.dump(truncated_curve, f)
        Mp_range = [None]
        # i_range = truncated_curve[:i_n]
        i_range = truncated_curve[:i_n]
    else:
        # The ranges for the different simulations
        i_range = np.linspace(90, 170, num=5) * u.deg
        Mp_range = np.array([1, 2, 20, 100, 300]) * u.M_earth

    # Constant values for all the planets
    w_p = 0 * u.rad
    W = 0 * u.rad
    Ms = 1 * u.M_sun
    p = 0.367
    RV_M0 = 0 * u.rad
    dist = 10 * u.pc

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5

    base_elements = (w_p, W, Ms, p, RV_M0, dist, fixed_K_val, IWA, OWA, dMag0)

    # Number of consistent orbits to generate
    n_fits = 50000

    # RV observation times
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = fun.gen_rv_times(initial_rv_time, final_rv_time)
    param_error = 0.1  # Also m/s

    ##########
    # Direct imaging observation times
    #########
    initial_obs_time = Time(2010.0, format="decimalyear")
    final_obs_time = Time(2030, format="decimalyear")
    timestep = 1 * u.d
    # timestep = 0.5*u.week
    # obs_n = 10
    # img_times = Time(np.linspace(initial_obs_time.jd, final_obs_time.jd, obs_n), format='jd')
    img_times = Time(
        np.arange(initial_obs_time.jd, final_obs_time.jd, step=timestep.to("d").value),
        format="jd",
    )

    print("Creating/checking chains")
    partial_chain_creation = functools.partial(
        chain_creation,
        rv_times,
        initial_rv_time,
        final_rv_time,
        param_error,
        base_elements,
        data_path,
    )
    element_list = list(product(a_range, i_range, e_range, Mp_range, rv_error_range))
    with ProcessPoolExecutor(max_workers=int(cores / 2.5)) as executor:
        results = executor.map(partial_chain_creation, element_list)

    planet_list = list(results)
    # Now that the chains have all been generated we can do the propagation

    rv_fit_options = {
        "n_fits": n_fits,
        "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
        "cov_samples": 1000,
        "fixed_inc": None,
        "fixed_f_sed": 3,
        "fixed_p": p,
    }

    rv_scatter_kwargs = {
        "color": "white",
        "alpha": 0.1,
        "s": 0.01,
        "label": "RV fit cloud",
    }
    rv_plot_kwargs = {
        "color": "white",
        "alpha": 1,
        "label": "RV fit cloud",
        "linewidth": 1,
        "linestyle": "-",
    }
    print("Creating populations")
    partial_population_creation = functools.partial(
        create_populations,
        dist,
        Ms,
        rv_fit_options,
        rv_plot_kwargs,
        rv_scatter_kwargs,
        img_times,
        dMag0,
        IWA,
        OWA,
        data_path,
        ["MG", "CI", "MKDE", "ML"],
    )
    with ProcessPoolExecutor(max_workers=cores) as executor:
        executor.map(partial_population_creation, planet_list)
    # for planet in planet_list:
    # # Debugging version
    # population = create_populations(dist, Ms, rv_fit_options, rv_plot_kwargs, rv_scatter_kwargs, img_times, dMag0, IWA, OWA,data_path, planet)

    # Calculate all the ratios
    ratios = []
    for planet in planet_list:
        ratio = planet.rv_error.decompose().value / planet.K.decompose().value
        ratios.append(round(ratio, 2))
    unique_ratios = np.unique(ratios)

    # Delete the folders
    types_of_plots = ["failure_plots"]
    # types_of_failure = ['no_failure', 'amplitude_failure', 'period_failure', 'intermittent_failure']
    types_of_failure = [
        "no_failure_(not_always_detectable)",
        "no_failure_(always_detectable)",
        "intermittent_failure",
        "dispersion_failure",
        "both_failure",
    ]
    fail_path = Path("plots", "failure_plots")
    if not fail_path.exists():
        os.mkdir(fail_path)
    for ratio in unique_ratios:
        directory = Path("plots/failure_plots/", f"{ratio:.2f}")
        directory = Path("plots/failure_plots/")
        for folder in directory.iterdir():
            if folder.is_dir():
                shutil.rmtree(folder)

    # Recreate the folders necessary
    for plot_type in types_of_plots:
        plot_path = Path("plots", plot_type)
        if not plot_path.exists():
            os.mkdir(plot_path)
        for ratio in unique_ratios:
            ratio_path = Path(plot_path, f"{ratio:.2f}")
            if not ratio_path.exists():
                os.mkdir(ratio_path)
            for failure_type in types_of_failure:
                failure_path = Path(ratio_path, failure_type)
                if not failure_path.exists():
                    os.mkdir(failure_path)

    # Defining colorscheme
    cmap = mpl.cm.get_cmap("viridis")
    colors = cmap(np.linspace(0, 0.95, 5))

    # Setting up the loop for the bar plots
    fig_bar, ((ml_ax, mkde_ax), (ci_ax, mg_ax)) = plt.subplots(
        ncols=2, nrows=2, figsize=[12.8205, 10]
    )
    # construction_methods = ['MG', 'CI', 'ML', 'MKDE']
    construction_method_dict = {
        "ml": (ml_ax, "Max likelihood", "_ML"),
        "mkde": (mkde_ax, "Max KDE", "_KDE"),
        "ci": (ci_ax, "Credible interval", "_CI"),
        "mg": (mg_ax, "Multivariate Gaussian", ""),
    }
    for method in construction_method_dict.keys():
        ax = construction_method_dict[method][0]
        method_name = construction_method_dict[method][1]
        method_suffix = construction_method_dict[method][2]
        # Keep track of failures
        planet_failure_modes = []
        list_of_ratios = []

        # array to keep track of the times of first intermittent failure
        first_if_list = []

        # This keeps track of the rate of intermittent failure for all that fail
        failure_rates = []
        for i, planet in enumerate(tqdm(planet_list)):
            failure_mode, failure_rates, first_if_list = failure_mode_calculations(
                data_path,
                planet,
                img_times,
                [IWA, OWA, dMag0],
                failure_rates,
                first_if_list,
                method_suffix=method_suffix,
            )
            # Store the failure information
            list_of_ratios.append(
                round(planet.rv_error.decompose().value / planet.K.decompose().value, 2)
            )
            planet_failure_modes.append(failure_mode)

        # Figure out the ratio breakdown of the failures
        u_ratios = np.unique(ratios)
        fail_df = pd.DataFrame(
            np.zeros((len(u_ratios), len(types_of_failure))),
            columns=types_of_failure,
            index=u_ratios,
        )
        for ratio in u_ratios:
            indices = np.where(list_of_ratios == ratio)[0]
            failures = np.array(planet_failure_modes)[indices]
            for fail in types_of_failure:
                fail_df.at[ratio, fail] = len(np.where(failures == fail)[0])

        print(fail_df)
        fail_df.to_pickle(Path(data_path, f"short_run_fail_df{method_suffix}.p"))
        # Create the tick and ticklabels so that the bars are evenly spaced
        x_ticks = np.linspace(0, 2, len(u_ratios))
        x_labels = [str(round(ratio, 2)) for ratio in u_ratios]
        # Create and save a bar plot with the different failure modes
        bar_path = Path(
            "plots/failure_plots/construction_method_bars_including_always_detectable.pdf"
        )
        # fig_bar, ax_bar = plt.subplots()
        percent_fail_df = 100 * fail_df / fail_df.sum(axis=1).values[0]
        types_of_failure_labels = [
            failure[0].upper() + failure[1:].replace("_", " ")
            for failure in types_of_failure
        ]
        for i, failure_type in enumerate(types_of_failure):
            failure_label = types_of_failure_labels[i]
            if i == 0:
                ax.bar(
                    x_ticks,
                    percent_fail_df[failure_type].to_numpy(),
                    0.17,
                    label=failure_label,
                    color=colors[-(i + 1)],
                )
                bottom_vals = percent_fail_df[failure_type].to_numpy()
            else:
                ax.bar(
                    x_ticks,
                    percent_fail_df[failure_type].to_numpy(),
                    0.17,
                    bottom=bottom_vals,
                    label=failure_label,
                    color=colors[-(i + 1)],
                )
                bottom_vals = (
                    bottom_vals + percent_fail_df[types_of_failure[i]].to_numpy()
                )

        # ax = fail_df.plot.bar(stacked=True)
        ax.set_xlabel("Ratio of error/semi-amplitude")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        # ax_bar.set_xticks(np.append(np.arange(0.25, 2.1, 0.25), 0.1))
        ax.set_ylabel("Percent")
        ax.set_title(
            f"{method_name}, planets per ratio: {len(planet_list)/len(u_ratios):.0f}"
        )
        # if ax.is_first_row() and ax.is_first_col():
        # # Only need one legend
        # # Want to flip the order to correspond to their order in the bar graph
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1], loc='lower right', framealpha=1, bbox_to_anchor=(1.3,0.6))
        # fig = ax.get_figure()
        if method_name == "Multivariate Gaussian":
            fail_df.to_csv(Path(data_path, "2500_run_MG_data.csv"))
    handles, labels = ax.get_legend_handles_labels()
    # fig_bar.legend(handles[::-1], labels[::-1], framealpha=1, loc=8, ncol=len(types_of_failure))
    # fig_bar.subplots_adjust(bottom=0.1)
    fig_bar.legend(handles[::-1], labels[::-1], framealpha=1, loc=7)
    fig_bar.subplots_adjust(right=0.78)
    # fig_bar.tight_layout()
    fig_bar.savefig(bar_path, dpi=400)


def violin_plot_style(ax, data, colors, u_ratios, annotate, ylims):
    """
    This is called to setup the style for the violin plots for the long multivariate Gaussian run
    """
    x_ticks = np.linspace(0, 2, len(u_ratios))
    x_labels = [str(round(ratio, 2)) for ratio in u_ratios]
    parts = ax.violinplot(
        data, x_ticks, widths=0.1, showmedians=False, showmeans=False, showextrema=False
    )
    # parts = ax.violinplot(data, u_ratios, widths = 0.1, showmedians=False, showmeans=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(colors[3])
        pc.set_edgecolor(colors[4])
        pc.set_alpha(0.35)
    quartile1, medians, quartile3 = (
        np.zeros(len(data)),
        np.zeros(len(data)),
        np.zeros(len(data)),
    )
    for i, _ in enumerate(data):
        quartile1[i], medians[i], quartile3[i] = np.percentile(data[i], [25, 50, 75])

    ax.scatter(x_ticks, medians, marker="_", color=colors[0], zorder=3)
    ax.vlines(
        x_ticks, quartile1, quartile3, color=colors[3], linestyle="-", lw=2, zorder=2
    )
    ax.set_ylim(ylims)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    if annotate:
        for i, _ in enumerate(u_ratios):
            # Want to label each one
            ax.annotate(
                f"n={len(data[i])}",
                (x_ticks[i], ylims[1]),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    return ax


def adjacent_values(vals, q1, q3):
    """
    This is used for the violin plots
    """
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def diffs_between_long_and_short(data_path):
    """This just calculates the difference between the long and short runs of
    the Multivariate Gaussian bar plot results"""
    # Get short run fail_df
    short_fail_path = Path(data_path, "short_run_fail_df.p")
    s_df = pd.read_pickle(short_fail_path)
    n_s = s_df.sum(axis=1).iloc[0]
    # Get long run fail_df
    long_fail_path = Path(data_path, "long_MG_fail_df.p")
    l_df = pd.read_pickle(long_fail_path)
    n_l = l_df.sum(axis=1).iloc[0]
    # Calculate difference
    diffs = np.abs(s_df / n_s - l_df / n_l) * 100
    print(f"Median: {np.median(diffs):.2f}%")
    print(f"Max: {diffs.stack().max():.2f}%")


def diffs_between_methods(data_path):
    methods = ["_ML", "_KDE", "_CI", ""]
    ratios = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 2]
    dfs = []
    for method in methods:
        method_df = pd.read_pickle(Path(data_path, f"short_run_fail_df{method}.p"))
        dfs.append(method_df)
    for i, ratio in enumerate(ratios):
        for j, df in enumerate(dfs):
            method = methods[j]
            no_failure_percentage = (
                100
                * (
                    df.loc[ratio, "no_failure_(not_always_detectable)"]
                    + df.loc[ratio, "no_failure_(always_detectable)"]
                )
                / df.sum(axis=1).iloc[0]
            )
            print(
                f"no_failure_percentage for {method} at ratio {ratio}: {no_failure_percentage}"
            )
        print("\n")


if __name__ == "__main__":
    computer = "tphon"
    # computer = "tphon"
    if computer == "home":
        d_path = "data"
        allowed_cores = 14
        base_inclinations = 100
        fix_a_value = True
    elif computer == "tphon":
        d_path = "/data/corey_data/rv_fitting"
        allowed_cores = 48
        base_inclinations = 250
        fix_a_value = False
    # orbits = error_in_time_comparisons(d_path)
    # time_vs_prob_det_both_failures(d_path)
    # error_in_time_comparisons(d_path)
    # consistent_orbits_error_comparisons(d_path)
    # consistent_orbits_error_comparisons_video(d_path)
    # consistent_orbit_comparison(d_path)
    # plotting_failures(d_path)
    # long_MG_bar_plot(d_path, allowed_cores, base_inclinations, fix_a_value)
    # short_all_construction_methods_bar_plot(
    # d_path, allowed_cores, base_inclinations, fix_a_value
    # )
    # time_vs_prob_det_both_failures(d_path)
    # diffs_between_long_and_short(d_path)
    # diffs_between_methods(d_path)
