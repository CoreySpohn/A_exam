import concurrent
import functools
import os
import pickle
import time
from pathlib import Path

import astropy.units as u
# import ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import radvel
from astropy.time import Time
from matplotlib.colors import ListedColormap
from planet import Planet
from planet_population import PlanetPopulation
from radvel.plot import mcmc_plots
from scipy import optimize
from scipy.interpolate import RectBivariateSpline, griddata, interp1d
from scipy.stats import gaussian_kde
from tqdm import tqdm


def gen_rv_times(initial_time, final_time, method=""):
    """
    This will create the times that the RV signal will be evaluated at giving options for different methods of calculation

    Args:
        initial_time:
            The first available observation time
        final_time:
            The last available observation time
        method:
            Value to determine how the times should be computed, currently not used
    """
    harps_path = Path("HARPS_data/GJ849.csv")
    harps_df = pd.read_csv(harps_path)
    obs_times = Time(harps_df["MJD-OBS"], format="mjd")
    obs_times_years = obs_times.decimalyear

    # Processing to get the times between 2004-2012 to get around the big gap
    main_years = obs_times_years[obs_times_years < 2012]

    # Calculate the number of observations per month
    months = (main_years[-1] - main_years[0]) * 12
    obs_per_month = len(main_years) / months

    # Find the number of months available for observation
    months_available = int((final_time.decimalyear - initial_time.decimalyear) * 12)

    # Calculate the observations per month for this period using a poisson process
    raw_schedule = np.random.poisson(lam=obs_per_month, size=months_available)

    # Go through the raw schedule and assign observations
    rv_times = np.zeros(sum(raw_schedule))
    obs_n = 0  # This keeps track of what the current observation is
    for month_num, obs_in_month in enumerate(raw_schedule):
        if obs_in_month == 0:
            continue
        # Get the month to assign values within
        time_period = initial_time + [month_num, month_num + 1] * u.yr / 12
        times_in_month = np.random.uniform(
            time_period[0].value, time_period[1].value, size=obs_in_month
        )
        for obs_time in times_in_month:
            # Add each observation in this month to the observation array
            rv_times[obs_n] = Time(obs_time, format="decimalyear").jd
            obs_n += 1
    rv_times[0] = initial_time.jd
    rv_times[-1] = final_time.jd
    return rv_times


def RV_data_plot(rv_data_df, error, planet):
    """
    Given the RV dataframe this will create a plot to show the signature is versus the real
    measurements/signal
    """
    # Get a clear RV curve to plot underneath at the true values
    model_times = Time(
        np.arange(rv_data_df["time"].iloc[0], rv_data_df["time"].iloc[-1], 2),
        format="jd",
    )
    true_curve = planet.calc_vs(model_times)
    plt.style.use("dark_background")
    # plt.rcParams['axes.labelsize']=25
    # plt.rcParams['axes.titlesize']=30
    # plt.rcParams['xtick.labelsize']=20
    # plt.rcParams['ytick.labelsize']=20
    fig, ax = plt.subplots(figsize=[5, 3])  # Was 5,3
    times = Time(rv_data_df.time.tolist(), format="jd").decimalyear
    ax.plot(model_times.decimalyear, true_curve, linewidth=1, color="white")
    ax.errorbar(
        times,
        rv_data_df["vel"],
        yerr=np.ones(len(rv_data_df["vel"])) * error,
        ecolor="red",
        fmt="o",
        markersize=3,
        color="cyan",
        markeredgecolor="black",
    )
    ax.set_title(f"Synthetic RV data, error: {error*100:.0f} cm/s")
    ax.set_xlabel("Time (year)")
    ax.set_ylabel("Radial velocity (m/s)")
    ax.set_ylim([-0.4, 0.4])
    fname = f"plots/rv_curve/rv_curve_{error:.2f}.png"
    plt.tight_layout()
    fig.savefig(fname, dpi=250)


def gen_radvel_params(planet, error):
    """
    This function will create the params necessary for radvel to fit the RV curve

    """
    # a, e, W, I, w_p = planet.orbElem
    e = planet.e
    w = planet.w_s.to(u.rad).value
    T_p = planet.T_p.jd
    T = planet.T.to(u.d).value
    K = planet.K.decompose().value

    #####################################################################
    # Fitting pararameters
    #####################################################################

    # Define global planetary system and dataset parameters
    # nplanets = 1  # number of planets in the system
    # list of instrument names. Can be whatever you like (no spaces) but should match 'tel' column in the input file.
    # instnames = ["i"]
    # ntels = len(instnames)  # number of instruments with unique velocity zero-points
    fitting_basis = "per tc secosw sesinw k"  # Fitting basis, see radvel.basis.BASIS_NAMES for available basis names
    # bjd0 = 0  # reference epoch for RV timestamps (i.e. this number has been subtracted off your timestamps)
    # map the numbers in the Parameters keys to planet letters (for plotting and tables)
    planet_letters = {1: "a"}

    # Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
    # initialize Parameters object
    anybasis_params = radvel.Parameters(
        1, basis="per tp e w k", planet_letters=planet_letters
    )

    # Create priors that aren't exact but are close
    anybasis_params["per1"] = radvel.Parameter(value=np.random.normal(T, T * error))

    # time of periapsis of 1st planet
    # In the case the standard deviation is a function of the period
    anybasis_params["tp1"] = radvel.Parameter(value=np.random.normal(T_p, T * error))

    # eccentricity of 1st planet
    anybasis_params["e1"] = radvel.Parameter(value=np.random.normal(e, e * error))

    # argument of periastron of the star's orbit for 1st planet
    anybasis_params["w1"] = radvel.Parameter(value=np.random.normal(w, w * error))

    # velocity semi-amplitude for 1st planet
    anybasis_params["k1"] = radvel.Parameter(value=np.random.normal(K, K * error))

    # slope: (If rv is m/s and time is days then [dvdt] is m/s/day)
    anybasis_params["dvdt"] = radvel.Parameter(value=0.0)

    # curvature: (If rv is m/s and time is days then [curv] is m/s/day^2)
    anybasis_params["curv"] = radvel.Parameter(value=0.0)

    # analytically calculate gamma if vary=False and linear=True
    # anybasis_params['gamma_i'] = radvel.Parameter(value=0, vary=False, linear=True)
    # anybasis_params['jit_i'] = radvel.Parameter(value=0)

    # Convert input orbital parameters into the fitting basis
    params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)

    # Set the 'vary' attributes of each of the parameters in the fitting basis. A parameter's 'vary' attribute should
    # be set to False if you wish to hold it fixed during the fitting process. By default, all 'vary' parameters
    # are set to True.
    # params['dvdt'].vary = False
    # params['curv'].vary = False

    return params


def gen_posterior(rv_data_df, params, error):
    """gen_chains.

    Args:
        rv_data_df:
        params:
    """
    time_base = np.mean(
        [np.min(rv_data_df.time), np.max(rv_data_df.time)]
    )  # abscissa for slope and curvature terms (should be near mid-point of time baseline)

    # Create the model and likilhood
    mod = radvel.RVModel(params, time_base=time_base)
    like = radvel.likelihood.RVLikelihood(
        mod,
        np.array(rv_data_df.time, dtype=float),
        np.array(rv_data_df.vel, dtype=float),
        np.array(rv_data_df.errvel, dtype=float),
    )
    # like.params["gamma"] = radvel.Parameter(value=0.0)
    # like.params["jit"] = radvel.Parameter(value=0.0)

    post = radvel.posterior.Posterior(like)
    # post.priors += [radvel.prior.Gaussian("jit", np.log(3), 0.5)]
    # post.priors += [radvel.prior.Gaussian("gamma", 0, 10)]
    # Set limit on the time of conjunctions so that the fits start when we want
    # Otherwise it defeats the purpose because propagating them back to the
    # times of direct imaging observation will spread them out
    post.priors += [radvel.prior.EccentricityPrior(1)]
    post.priors += [
        radvel.prior.Gaussian("tc1", params["tc1"].value, params["per1"].value * error)
    ]
    # post.priors += [radvel.prior.HardBounds('tc1', params['tc1'].value - params['per1'].value, params['tc1'].value + params['per1'].value)]
    post.priors += [
        radvel.prior.Gaussian(
            "per1", params["per1"].value, params["per1"].value * error
        )
    ]
    # post.priors += [radvel.prior.HardBounds('per1', params['per1'].value - params['per1'].value/10, params['per1'].value + params['per1'].value/10)]

    res = optimize.minimize(
        post.neglogprob_array, post.get_vary_params(), method="Nelder-Mead"
    )
    return post


def gen_chains(post):
    """TODO: Docstring for gen_chains.

    :function: TODO
    :returns: TODO

    """
    while True:
        # This is here because some intial conditions result in the fit failing
        attempt_num = 0
        try:
            chains = radvel.mcmc(post, nrun=10000, ensembles=3)
            break
        except:
            attempt_num += 1
            print(f"Retrying RadVel fitting, attempt {attempt_num}...")
    return chains


def time_plot(
    planet_list,
    times,
    dMag0,
    IWA,
    OWA,
    error_match,
    time_back,
    photdict,
    bandzip,
    kde_elements=None,
):
    """time_plot.

    Args:
        orbElem:
        M0:
        Mp:
        Ms:
        p:
        Rp:
        planets:
        dist:
        times:
        dMag0:
    """
    times_progressed = []
    # Calculate the working angle contrasts
    WAs = np.linspace(IWA, OWA, 1000)
    WA_contrasts = np.ones(1000) * 10 ** (dMag0 / -2.5)

    # IWA = IWA.to(u.arcsec).value
    # OWA = OWA.to(u.arcsec).value

    plt.style.use("dark_background")
    for i, t in enumerate(tqdm(times, desc="Separation vs flux plots: ")):
        # Separation flux and completeness plots
        fig, (fs_ax, comp_ax) = plt.subplots(2, figsize=(7.111, 4))
        # Set up the plots so that the legend is outside
        fs_box = fs_ax.get_position()
        fs_ax.set_position([fs_box.x0, fs_box.y0, fs_box.width, fs_box.height])
        comp_box = comp_ax.get_position()
        comp_ax.set_position(
            [comp_box.x0, comp_box.y0, comp_box.width, comp_box.height]
        )
        comp_ax.set_xlim(times.decimalyear[0], times.decimalyear[-1])
        comp_ax.set_ylim([-0.10, 1.1])
        comp_ax.set_yticks(np.linspace(0, 1, num=6))
        comp_ax.set_ylabel("Probability of detection")
        comp_ax.set_xlabel("Year")
        fs_ax.set_ylim([10 ** -12, 10 ** -6])
        fs_ax.set_xlim([0, 0.3])
        times_progressed.append(Time(t, format="jd").decimalyear)
        fs_ax.plot(
            WAs,
            WA_contrasts,
            alpha=0.85,
            label="Telescope limits",
            linewidth=1,
            color="orange",
            linestyle="--",
        )
        fs_ax.set_title(f"{t.decimalyear:.2f}")
        fs_ax.set_xlabel('Separation (")')
        fs_ax.set_ylabel("Flux Ratio")
        fs_ax.set_yscale("log")
        # Defining a colormap for the cloud
        cmap = plt.cm.inferno
        my_cmap = cmap(np.arange(cmap.N))
        # set alpha for last element
        # my_cmap[:,-1] = np.linsspace(0,1,cmap.N)
        # my_cmap[:128,-1] = np.zeros(128)
        my_cmap[:32, -1] = np.zeros(32)
        my_cmap = ListedColormap(my_cmap)
        for j, current_planet in enumerate(planet_list):
            if not hasattr(current_planet, "percent_detectable"):
                current_planet.percent_detectable = []
            WA, dMag, beta = current_planet.prop_for_imaging(t, photdict, bandzip)

            # find the different visibility measures
            visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)

            # Compute percents
            current_planet.percent_detectable.append(sum(visible) / current_planet.num)

            # Top subplot of separation vs flux
            separation_for_plot = WA.to(u.arcsec)
            flux_ratio = 10 ** (dMag / -2.5)
            if isinstance(current_planet, PlanetPopulation):
                xx, yy = np.mgrid[
                    min(separation_for_plot) : max(separation_for_plot) : 100j,
                    min(flux_ratio) : max(flux_ratio) : 100j,
                ]
                # xx, yy = np.mgrid[fs_ax.get_xlim()[0]:fs_ax.get_xlim()[1]:50j, fs_ax.get_ylim()[0]:fs_ax.get_ylim()[1]:100j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([separation_for_plot.value, flux_ratio])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                cfset = fs_ax.contourf(xx, yy, f, levels=100, cmap=my_cmap)
                fig.colorbar(cfset, ax=fs_ax)
                # fs_ax.imshow(alpha=0.2, extent = fs_ax.get_xlim() +fs_ax.get_ylim)
                # cset = fs_ax.contour(xx, yy, f, colors='k')
                # fs_ax.clabel(cset, inline=1, fontsize=10)
            else:
                fs_ax.scatter(
                    separation_for_plot,
                    flux_ratio,
                    **current_planet.scatter_kwargs,
                )

            # Bottom subplot with the completness
            comp_ax.plot(
                times_progressed,
                current_planet.percent_detectable,
                **current_planet.plot_kwargs,
            )

        comp_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
        fs_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
        fig.tight_layout()
        fname = f"plots/time_evolving/sep_flux/sep_flux{i:04}.png"
        fs_ax.figure.savefig(fname, dpi=250)
        plt.close()

    # Creating the error plot
    # Start by gettin the true planet's visibility array and separate the fitted planets
    fitted_planets = []
    for current_planet in planet_list:
        current_planet.visibility_array = np.array(current_planet.percent_detectable)
        if current_planet.planet_label == "True planet":
            true_visibility = current_planet.visibility_array
        else:
            fitted_planets.append(current_planet)

    # Create the figure info
    fig, err_ax = plt.subplots(1, figsize=(7.111, 4))
    # err_ax.set_title(f'Percent of time the fit accuratly predicts visibility to {100*error_match:.1f}%')
    # err_ax.set_ylabel(f'Percent of the last {time_back.to(u.yr).value:.0f} years')
    err_ax.set_ylabel("Difference in probability of detection from true planet")
    err_ax.set_xlabel("Year")
    err_ax.set_xticks(np.arange(times.decimalyear[0], times.decimalyear[-1], 2.5))
    err_box = err_ax.get_position()
    err_ax.set_position([err_box.x0, err_box.y0, err_box.width * 0.8, err_box.height])

    # Now check each of the other planets to see how close they match the true planet
    for current_planet in fitted_planets:
        current_planet.visibility_match = np.abs(
            true_visibility - current_planet.visibility_array
        )
        current_planet.recent_match_list = []

    # List of the times that will have meaningful results since they're past the time_back period
    # relevant_times_for_plot = []

    # Now look through the times and plot them
    # for i, t in enumerate(tqdm(times, desc="Error plots: ")):
    # # Find the index values that are within the time_back period
    # relative_times = times.decimalyear-t.decimalyear
    # time_back_indices = np.where((relative_times<0) & (relative_times>-time_back.to(u.yr).value))[0]
    # # Note that the [0] is to get the array out of the tuple that numpy returns

    # if ( t.decimalyear-times[0].decimalyear ) < time_back.to(u.yr).value:
    # # Don't find the percentages for the periods before the full average time
    # continue
    # else:
    # relevant_times_for_plot.append(t.decimalyear)

    # # Now have to assign the values to an array to plot
    # for current_planet in fitted_planets:
    # # This is the visibility match at the times within the time period set by time_back
    # recent_match_percents = current_planet.visibility_match[time_back_indices]

    # # Find how many of these times are within the error_match tolerance
    # match_to_error = np.where(error_match>recent_match_percents)[0]
    # percent_matching = len(match_to_error)/len(recent_match_percents)
    # # Calculate the average agreement
    # current_planet.recent_match_list.append(percent_matching)

    # Now plot all the stuff
    for current_planet in fitted_planets:
        # err_ax.plot(relevant_times_for_plot, current_planet.recent_match_list, **current_planet.plot_kwargs)
        err_ax.plot(
            times.decimalyear,
            current_planet.visibility_match,
            **current_planet.plot_kwargs,
        )
    err_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
    fname = "plots/time_evolving/error.png"
    err_ax.figure.savefig(fname, dpi=250)
    plt.close()


def time_comparison_plot(
    planet_list, times, dMag0, IWA, OWA, photdict, bandzip, kde_elements=None
):
    """
    This plot will show how the fits spread out in time between two points
    """
    plt.style.use("dark_background")
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    WAs = np.linspace(IWA, OWA, 1000)
    WA_contrasts = np.ones(1000) * 10 ** (dMag0 / -2.5)
    normi = mpl.colors.Normalize(vmin=0, vmax=100000)
    # levels = np.linspace(0,1,101)
    for i, t in enumerate(times):
        fig, ax = plt.subplots(figsize=(10.64, 7))
        ax.set_ylim([10 ** -10, 10 ** -8])
        ax.set_xlim([0, 0.25])
        # ax.plot(
        # WAs,
        # WA_contrasts,
        # alpha=0.85,
        # label="Telescope limits",
        # linewidth=1,
        # color="orange",
        # linestyle="--",
        # )
        ax.set_title(f"{t.decimalyear:.2f}, after {i} periods")
        ax.set_xlabel('Separation (")')
        ax.set_ylabel("Flux Ratio")
        ax.set_yscale("log")
        cmap = plt.cm.inferno
        my_cmap = cmap(np.arange(cmap.N))
        my_cmap[:2, -1] = np.zeros(2)
        my_cmap = ListedColormap(my_cmap)
        for current_planet in planet_list:
            WA, dMag, beta = current_planet.prop_for_imaging(t, photdict, bandzip)

            # find the different visibility measures
            visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)

            # Top subplot of separation vs flux
            separation_for_plot = WA.to(u.arcsec)
            flux_ratio = 10 ** (dMag / -2.5)
            if isinstance(current_planet, PlanetPopulation):
                xx, yy = np.mgrid[
                    0.75
                    * min(separation_for_plot) : 1.25
                    * max(separation_for_plot) : 250j,
                    0.5 * min(flux_ratio) : 1.5 * max(flux_ratio) : 250j,
                ]
                # xx, yy = np.mgrid[ax.get_xlim()[0]:ax.get_xlim()[1]:500j, ax.get_ylim()[0]:ax.get_ylim()[1]:500j]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                values = np.vstack([separation_for_plot.value, flux_ratio])
                kernel = gaussian_kde(values)
                f = np.reshape(kernel(positions).T, xx.shape)
                scaled_f = f / max(map(max, f))
                # print(max(map(max,f)))
                cfset = ax.contourf(xx, yy, scaled_f, levels=100, cmap=my_cmap)
                # cbar = fig.colorbar(cfset, ax=ax, ticks=np.linspace(0,1,11))
                # fs_ax.imshow(alpha=0.2, extent = fs_ax.get_xlim() +fs_ax.get_ylim)
                # cset = fs_ax.contour(xx, yy, f, colors='k')
                # fs_ax.clabel(cset, inline=1, fontsize=10)
            else:
                ax.scatter(
                    separation_for_plot,
                    flux_ratio,
                    **current_planet.scatter_kwargs,
                )
        # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
        fig.tight_layout()
        fname = f"plots/time_evolving/time_comparision/time_{i:02}.png"
        ax.figure.savefig(fname, dpi=250)
        plt.close()


def _prop_for_imaging(pop, dMag0, IWA, OWA, t):
    """
    Method to be run that finds the probability of detection for a set time, orbit fit, and coronagraph parameters
    """
    WA, dMag, beta = pop.prop_for_imaging(t)
    visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
    percent_detectable = sum(visible) / pop.num
    # print(f'time {t}: {percent_detectable}')
    # print("hello from {}".format(os.getpid()))
    return (t, percent_detectable)


def _prop_for_imaging_c(pop, dMag0, IWA, OWA, t):
    """
    Method to be run that finds the probability of detection for a set time, orbit fit, and coronagraph parameters
    """
    WA, dMag, beta = pop.prop_for_imaging_c(t)
    visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
    percent_detectable = sum(visible) / pop.num
    # print(f'time {t}: {percent_detectable}')
    # print("hello from {}".format(os.getpid()))
    return (t, percent_detectable)


def time_until_failure(
    orbit_fit, true_planet, prob_thresh, failure_duration, times, dMag0, IWA, OWA
):
    # Find the first time that the planet is above the threshold of failure longer than the failure_duration
    orbit_fit.failure_time_indicies = []
    orbit_fit.failure_times = []
    orbit_fit.consecutive_fails = []
    orbit_fit.percent_detectable = []
    found_first_failure = False
    time_step = (times[1] - times[0]).jd * u.d
    # orbit_fit.first_failure = np.nan

    # Multiprocessing the propagation to speed up code
    # This line makes it so I can feed in multiple inputs easily
    partial_prop_for_imaging = functools.partial(
        _prop_for_imaging, orbit_fit, dMag0, IWA, OWA
    )
    partial_prop_for_imaging_c = functools.partial(
        _prop_for_imaging_c, orbit_fit, dMag0, IWA, OWA
    )

    # print(f'Calculating probability of detection for planet {true_planet.planet_label}')

    # Now set up the process pool to actually run everything
    # t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
        results = executor.map(partial_prop_for_imaging, times)
    # tf = time.perf_counter()
    # print(f'Regular parallel: {tf-t0}')
    # t0 = time.perf_counter()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    # results = executor.map(partial_prop_for_imaging_c, times)
    # tf = time.perf_counter()
    # print(f'C parallel: {tf-t0}')

    # Sorting the results and storing them in class parameters
    # rs = list(zip(*list(results)))
    # orbit_fit.pd_times = rs[0]
    # orbit_fit.percent_detectable = rs[1]

    # After calculating the probability of detection, go back to analyze it for the time until failure
    # t0 = time.perf_counter()
    # for i, t in enumerate(tqdm(times, desc="Probability of detection, Python")):
    # WA, dMag, beta = orbit_fit.prop_for_imaging(t)
    # visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
    # orbit_fit.percent_detectable.append(sum(visible)/orbit_fit.num)
    # tf = time.perf_counter()
    # print(orbit_fit.percent_detectable)
    # print(f'Regular time: {tf-t0}')
    # print('\n')

    # t0 = time.perf_counter()
    # orbit_fit.percent_detectable = []
    # for i, t in enumerate(tqdm(times, desc="Probability of detection, C")):
    # WA, dMag = orbit_fit.prop_for_imaging_c(t)
    # visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)
    # orbit_fit.percent_detectable.append(sum(visible)/orbit_fit.num)
    # tf = time.perf_counter()
    # print(orbit_fit.percent_detectable)
    # print(f'C time: {tf-t0}')
    # print('\n')

    # t0 = time.perf_counter()
    # cython_pd = orbit_fit.prob_det(times, IWA, OWA, dMag0)
    # tf = time.perf_counter()
    # print(cython_pd)
    # print(f'C loop time: {tf-t0}')

    print("\n")
    t0 = time.perf_counter()
    pure_c_pd = orbit_fit.prob_det_pc(times, IWA, OWA, dMag0)
    tf = time.perf_counter()
    print(pure_c_pd)
    print(f"Pure c time: {tf-t0}")
    print("\n")

    # per_d = orbit_fit.percent_detectable[i]
    # if per_d >= prob_thresh:
    # # print(f'Above threshold at {t.decimalyear}')
    # WA_t, dMag_t, beta_t = true_planet.prop_for_imaging(t)
    # is_visible = (IWA < WA_t) & (OWA > WA_t) & (dMag0 > dMag_t)
    # if is_visible == 1:
    # continue
    # else:
    # # Check to see whether the threshold has been met
    # if (orbit_fit.failure_times != []) and (orbit_fit.failure_time_indicies[-1] == i-1):
    # # If the last failure was the previous time checked then iterate the consecutive_fails
    # consecutive_fails += 1
    # else:
    # consecutive_fails = 1
    # orbit_fit.consecutive_fails.append(consecutive_fails)
    # orbit_fit.failure_times.append(t)
    # orbit_fit.failure_time_indicies.append(i)
    # if (consecutive_fails*time_step>= failure_duration) and not found_first_failure:
    # found_first_failure = True
    # orbit_fit.first_failure = t


def point_cloud_comparison(
    planet_list, times, dMag0, IWA, OWA, rv_error, kde_elements=None
):
    times_progressed = []
    # Calculate the working angle contrasts

    # plt.style.use("dark_background")
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    # plt.rcParams['legend.fontsize']=12
    save_all_plots = False
    for i, t in enumerate(tqdm(times, desc="Probability of detection")):
        # Separation flux and completeness plots
        times_progressed.append(Time(t, format="jd").decimalyear)
        # Defining a colormap for the cloud
        for j, current_planet in enumerate(planet_list):
            if not hasattr(current_planet, "percent_detectable"):
                current_planet.percent_detectable = []
            WA, dMag = current_planet.prop_for_imaging(t)

            # find the different visibility measures
            visible = (IWA < WA) & (OWA > WA) & (dMag0 > dMag)

            # Compute percents
            current_planet.percent_detectable.append(sum(visible) / current_planet.num)

    # Now make the plot
    # fig, comp_ax = plt.subplots(figsize=(12, 5)) # was 10,4
    fig, comp_ax = plt.subplots(figsize=(10, 4))  # was 10,4
    # Set up the plots so that the legend is outside
    comp_box = comp_ax.get_position()
    comp_ax.set_position([comp_box.x0, comp_box.y0, comp_box.width, comp_box.height])
    comp_ax.set_xlim(times.decimalyear[0], times.decimalyear[-1])
    comp_ax.set_ylim([-0.10, 1.1])
    comp_ax.set_yticks(np.linspace(0, 1, num=6))
    comp_ax.set_ylabel("Probability of detection")
    comp_ax.set_xlabel("Year")
    for current_planet in planet_list:
        # Bottom subplot with the completness
        comp_ax.plot(
            times_progressed,
            current_planet.percent_detectable,
            **current_planet.plot_kwargs,
        )
    # comp_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    comp_ax.legend(loc="lower left")
    comp_ax.set_title(f"RV error: {rv_error:.1f}")
    fig.tight_layout()
    fname = "plots/time_evolving/prob_det/prob_det.png"
    fig.savefig(fname, dpi=250)
    plt.close()


def load_phot_data(filename="allphotdata_2015.npz"):
    """load_phot_data.

    Args:
        filename:
    """
    tmp = np.load(filename)
    allphotdata = tmp["allphotdata"]
    clouds = tmp["clouds"]
    cloudstr = tmp["cloudstr"]
    wavelns = tmp["wavelns"]
    betas = tmp["betas"]
    dists = tmp["dists"]
    metallicities = tmp["metallicities"]

    def makeninterp(vals):
        """makeninterp.

        Args:
            vals:
        """
        ii = interp1d(
            vals,
            vals,
            kind="nearest",
            bounds_error=False,
            fill_value=(vals.min(), vals.max()),
        )
        return ii

    distinterp = makeninterp(dists)
    betainterp = makeninterp(betas)
    feinterp = makeninterp(metallicities)
    cloudinterp = makeninterp(clouds)

    photinterps2 = {}
    quadinterps = {}
    for i, fe in enumerate(metallicities):
        photinterps2[fe] = {}
        quadinterps[fe] = {}
        for j, d in enumerate(dists):
            photinterps2[fe][d] = {}
            quadinterps[fe][d] = {}
            for k, cloud in enumerate(clouds):
                if np.any(np.isnan(allphotdata[i, j, k, :, :])):
                    # remove whole rows of betas
                    goodbetas = np.array(
                        list(
                            set(range(len(betas)))
                            - set(
                                np.unique(
                                    np.where(np.isnan(allphotdata[i, j, k, :, :]))[0]
                                )
                            )
                        )
                    )
                    photinterps2[fe][d][cloud] = RectBivariateSpline(
                        betas[goodbetas], wavelns, allphotdata[i, j, k, goodbetas, :]
                    )
                    # photinterps2[fe][d][cloud] = interp2d(betas[goodbetas],wavelns,allphotdata[i,j,k,goodbetas,:].transpose(),kind='cubic')
                else:
                    # photinterps2[fe][d][cloud] = interp2d(betas,wavelns,allphotdata[i,j,k,:,:].transpose(),kind='cubic')
                    photinterps2[fe][d][cloud] = RectBivariateSpline(
                        betas, wavelns, allphotdata[i, j, k, :, :]
                    )
                quadinterps[fe][d][cloud] = interp1d(
                    wavelns, allphotdata[i, j, k, 9, :].flatten()
                )

    return {
        "allphotdata": allphotdata,
        "clouds": clouds,
        "cloudstr": cloudstr,
        "wavelns": wavelns,
        "betas": betas,
        "dists": dists,
        "metallicities": metallicities,
        "distinterp": distinterp,
        "betainterp": betainterp,
        "feinterp": feinterp,
        "cloudinterp": cloudinterp,
        "photinterps": photinterps2,
        "quadinterps": quadinterps,
    }


def gen_bands():
    """
    Generate central wavelengths and bandpasses describing bands of interest
    """

    lambdas = [575, 660, 730, 760, 825]  # nm
    bps = [10, 18, 18, 18, 10]  # percent
    bands = []
    bandws = []
    bandwsteps = []

    for lam, bp in zip(lambdas, bps):
        band = np.array([-1, 1]) * float(lam) / 1000.0 * bp / 200.0 + lam / 1000.0
        bands.append(band)
        [ws, wstep] = np.linspace(band[0], band[1], 100, retstep=True)
        bandws.append(ws)
        bandwsteps.append(wstep)

    bands = np.vstack(bands)  # um
    bws = np.diff(bands, 1).flatten()  # um
    bandws = np.vstack(bandws)
    bandwsteps = np.array(bandwsteps)

    return zip(lambdas, bands, bws, bandws, bandwsteps)


def delete_pictures_in_path(path):
    for file in os.scandir(path):
        os.unlink(file.path)


def gen_videos(input_path, output_path, framerate):
    (ffmpeg.input(input_path, framerate=framerate).output(output_path).run())


if __name__ == "__main__":
    # The actual parameters of the system
    true_a = 1.5 * u.AU
    true_e = 0.198
    true_w_s = 1 * u.rad  # This is the star's, to get the planet add 180 deg
    true_w_p = (true_w_s + np.pi * u.rad) % (2 * np.pi * u.rad)
    true_W = 1 * u.rad
    true_I = np.pi / 2 * u.rad
    true_Mp = 1.5 * u.M_earth
    # true_Rp = 1.5 * u.R_earth
    true_Rp = Planet.RfromM(None, true_Mp)[0]
    true_Ms = 1 * u.M_sun
    true_p = 0.37
    true_f_sed = [3]
    true_fe = [1]
    true_orbElem = (true_a, true_e, true_W, true_I, true_w_p)
    true_RV_M0 = 0 * u.rad
    # This is the mean anomaly at the time of the first RV measurment
    dist = 10 * u.pc

    # This is the number of samples to take for the planet populations
    n_fits = 10000

    #######
    # Times of RV observations
    #######
    initial_rv_time = Time(2000.0, format="decimalyear")
    final_rv_time = Time(2010.0, format="decimalyear")
    rv_times = gen_rv_times(initial_rv_time, final_rv_time)
    rv_error = 0.1  # This is in m/s

    ##########
    # Direct imaging observation times
    #########
    initial_obs_time = Time(2010.0, format="decimalyear")
    final_obs_time = Time(2030, format="decimalyear")
    img_observations = 500
    percent_detectable = np.zeros(img_observations)
    img_times = Time(
        np.linspace(initial_obs_time.jd, final_obs_time.jd, img_observations),
        format="jd",
    )

    # This is used to tune how close to the real value the guesses are
    param_error = 0.1

    ######
    # Telescope parameters
    ######
    # Roman parameters
    # IWA = 0.19116374 * u.arcsec
    # OWA = 0.57349123 * u.arcsec
    # dMag0 = 20.5

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 10 * u.arcsec
    dMag0 = 26.5
    metallicities = [1]

    ########
    # Plot values
    ########
    error_match = 0.1
    time_back = 2 * u.yr

    ########
    # Plot kwargs
    ########
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
        "color": "lime",
        "alpha": 1,
        "linewidth": 1,
        "label": "True planet",
        "linestyle": "dashdot",
    }

    # kwargs used for the actual RV fits
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
        "color": "blue",
        "alpha": 1,
        "label": "Credible interval",
        "markersize": 5,
        "linestyle": "dashed",
        "linewidth": 1,
    }

    # Maximum likelihood kwargs
    ml_scatter_kwargs = {
        "color": "red",
        "alpha": 1,
        "s": 10,
        "label": "Max likelihood",
        "marker": "x",
        "edgecolors": "white",
    }
    ml_plot_kwargs = {
        "color": "red",
        "alpha": 1,
        "label": "Max likelihood",
        "markersize": 5,
        "linestyle": "dotted",
        "linewidth": 1,
    }

    # Kernel density estimate kwargs
    kde_scatter_kwargs = {
        "color": "magenta",
        "alpha": 1,
        "s": 10,
        "label": "Max KDE",
        "marker": "*",
        "edgecolors": "white",
        "linewidth": 0.5,
    }
    kde_plot_kwargs = {
        "color": "magenta",
        "alpha": 1,
        "label": "Max KDE",
        "markersize": 5,
        "linewidth": 1,
        "linestyle": (0, (5, 1)),  # "linestyle": (0, (5, 10)),
    }

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
    }

    # Cram all the information into the class
    true_planet = Planet(
        "True planet",
        dist,
        true_Ms,
        true_fe,
        true_plot_kwargs,
        true_scatter_kwargs,
        keplerian_inputs=true_inputs,
    )

    #######
    # Create the path locations
    #######
    base_path = true_planet.gen_base_path(rv_error, initial_rv_time, final_rv_time)
    rv_curve_path = Path("data", "rv_curve", base_path).with_suffix(".csv")
    post_path = Path("data", "post", base_path).with_suffix(".p")
    chains_path = Path("data", "chains", base_path).with_suffix(".csv")
    covariance_path = Path("data", "covariance", base_path).with_suffix(".p")

    photdict = load_phot_data()
    bandinfo = list(gen_bands())[0]

    if rv_curve_path.exists():
        rv_df = pd.read_csv(rv_curve_path).drop(columns=["Unnamed: 0"])
    else:
        rv_df = true_planet.simulate_rv_observations(rv_times, rv_error)
        # rv_df = RV_data_gen(true_planet, rv_times, rv_error)
        rv_df.to_csv(rv_curve_path)

    if chains_path.exists():
        chains = pd.read_csv(chains_path).drop(columns=["Unnamed: 0"])
        post = pickle.load(open(post_path, "rb"))
    else:
        params = gen_radvel_params(true_planet, param_error)
        post = gen_posterior(rv_df, params, param_error)
        chains = gen_chains(post)
        pickle.dump(post, open(post_path, "wb"))
        chains.to_csv(chains_path)

    ###############
    # Create the population of planets using the chains
    ###############
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
    if not covariance_path.exists():
        pickle.dump(rv_fits.cov_df, open(covariance_path, "wb"))
    # Create the credible interval planet
    ci_chain = rv_fits.chains.quantile([0.5])
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
    ci_std = rv_fits.chains.std()
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
    kde = rv_fits.get_kde_estimate()
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

    # Set up paths
    save_location = Path("plots/time_evolving")
    sep_flux_pic_path = Path(save_location, "sep_flux/sep_flux%04d.png")
    sep_flux_vid_path = Path(save_location, "sep_flux.mp4")

    # Remove pictures from the last run
    delete_pictures_in_path(Path(save_location, "sep_flux"))

    # Create the fun plots
    # This plot will show the cloud at multiple periods
    # Basically have to find one time when the true_planet is at a specific
    # mean anomaly and then add the period to it over and over again
    comparision_times = np.arange(img_times[0], img_times[-1], true_planet.T)
    time_comparison_plot(
        [rv_fits, true_planet], comparision_times, dMag0, IWA, OWA, photdict, bandinfo
    )

    # Create the plot showing the different probability of detections
    initial_comp_time = Time(2010.0, format="decimalyear")
    final_comp_time = Time(2030, format="decimalyear")
    comp_observations = 2
    comp_times = Time(
        np.linspace(initial_comp_time.jd, final_comp_time.jd, comp_observations),
        format="jd",
    )
    point_cloud_comparison(
        [
            true_planet,
            rv_fits,
            ci_planet_population,
            ml_planet_population,
            kde_planet_population,
        ],
        comp_times,
        dMag0,
        IWA,
        OWA,
        photdict,
        bandinfo,
        rv_error,
    )

    # time_plot(
    # [rv_fits, true_planet, ci_planet, ml_planet, kde_planet],
    # img_times,
    # dMag0,
    # IWA,
    # OWA,
    # error_match,
    # time_back,
    # photdict,
    # bandinfo,
    # )

    # Create the video
    # gen_videos(str(sep_flux_pic_path), str(sep_flux_vid_path), 25)
