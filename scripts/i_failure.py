import numpy as np
from tqdm import tqdm
import pandas as pd
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from planet_population import PlanetPopulation
from pathlib import Path
import pickle
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import functions as fun
import paper as pap
import astropy.constants as const

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))


if __name__ == '__main__':
    rv_times = Time(np.linspace(2000, 2004, 100), format='decimalyear')
    # Dispersion failure
    failure = 'i_failure'
    if failure == 'n_failure':
        img_times = Time(np.linspace(2004, 2005, 365), format='decimalyear')
        pop_types = ['CI']
        rv_error = 0.5*u.m/u.s
        a = 1*u.AU
    elif failure == 'i_failure':
        img_times = Time(np.linspace(2004, 2005, 365), format='decimalyear')
        pop_types = ['ML']
        rv_error = 0.5*u.m/u.s
        a = 1*u.AU
    elif failure == 'd_failure':
        img_times = Time(np.linspace(2004, 2010, 1000), format='decimalyear')
        pop_types = ['CI']
        rv_error = 1*u.m/u.s
        a = 1.5*u.AU
    data_path = 'data'
    n_fits = 50000

    p = 0.367
    K_val = 1*u.m/u.s
    p1_inputs = {'a': a,
                 'e': 0,
                 'W': 0*u.rad,
                 'I': 90*u.degree,
                 'w': 0*u.rad,
                 'Mp': 1*u.M_earth,
                 'Rp': 1*u.R_earth,
                 'f_sed': 3,
                 'p': p,
                 'M0': 0*u.rad,
                 't0': Time(rv_times[0], format='decimalyear'),
                 'rv_error': rv_error}
    ratio=p1_inputs['rv_error']/K_val
    p1 = Planet(r'1 $M_\oplus$', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p1_inputs)
    Mp = (
        K_val
        * (1*u.M_sun) ** (2 / 3)
        * np.sqrt(1 - p1.e ** 2)
        / np.sin(p1.I.to(u.rad))
        * (p1.T / (2 * np.pi * const.G)) ** (1 / 3)
    ).decompose().to(u.M_earth)
    Rp = p1.RfromM(Mp.to(u.M_earth).value)
    p1_inputs['Mp'] = Mp
    p1_inputs['Rp'] = Rp[0]

    p1 = Planet(r'1 $M_\oplus$', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p1_inputs)

    rv_df = p1.simulate_rv_observations(Time(rv_times, format='decimalyear'), p1.rv_error)

    font = {'size': 13}
    plt.rc("font", **font)
    plt.style.use("dark_background")
    # Make RV plot
    fig, rv_ax = plt.subplots(figsize=(12,6))

    current_p1_rvs = rv_df['vel']
    current_p1_times = rv_df['time']
    norm = mpl.colors.Normalize(vmin=min(rv_df['truevel']), vmax=max(rv_df['truevel']))
    my_cmap = plt.get_cmap('coolwarm')

    rv_ax.errorbar(Time(list(current_p1_times), format='jd').decimalyear, current_p1_rvs, yerr=p1.rv_error.decompose().value, alpha=0.5, fmt='none', zorder= 1)
    rv_ax.scatter(Time(list(current_p1_times), format='jd').decimalyear, current_p1_rvs, c=current_p1_rvs, cmap=my_cmap, norm=norm, zorder=2)
    rv_ax.set_xlim([rv_times[0].value, rv_times[-1].value])
    rv_ax.set_ylim([-1, 1])
    rv_ax.set_xlabel('Year')
    rv_ax.set_ylabel('RV (m/s)')
    rv_ax.axhline(0, color='white', linestyle='--', alpha = 0.5)
    # rv_ax.annotate(f'$K$ = {max(rv_df["truevel"]):.3f} (m/s)', xytext=(2002, 0.01), xy=(2002, min(rv_df['truevel'])), ha='center', va='center', arrowprops=dict(arrowstyle='->'), zorder=10)
    # rv_ax.annotate(f'Ratio = {p1.rv_error.value/max(rv_df["truevel"]):.2f}', xy=(2002, -.09), ha='center', va='center')
    rv_ax.set_title(f'Ratio = {p1.rv_error.value/max(rv_df["truevel"]):.2f}', ha='center', va='center')
    fig.tight_layout()
    fig.savefig(Path(f'../figures/failure_RV_curve-{pop_types[0]}-ratio-{ratio:.2f}.png'), dpi=150)

    #### HabEx starshade performance
    IWA = 0.058 * u.arcsec
    OWA = 6 * u.arcsec
    dMag0 = 26.5
    chain_elements = (p1.a, p1.I, p1.e, p1.Mp, p1.rv_error, p1.Rp)
    constant_elements = (p1.w, p1.W, p1.Ms, p1.p, 0*u.rad, p1.dist_to_star, None, IWA, OWA, dMag0)
    param_error = 0.1
    fit_options = {
        "n_fits": n_fits,
        "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
        "cov_samples": 1000,
        "fixed_inc": None,
        "fixed_f_sed": 3,
        "fixed_p": p,
    }

    base_path = p1.gen_base_path(p1.rv_error, rv_times[0], rv_times[-1], IWA, OWA, dMag0)
    print(pop_types[0])
    pap.chain_creation(rv_times, rv_times[0], rv_times[-1], param_error, constant_elements, data_path, chain_elements)
    pap.create_populations(p1.dist_to_star, p1.Ms, fit_options, {}, {}, img_times, dMag0, IWA, OWA, data_path, pop_types, p1)
    # base_path = p1.gen_base_path(
        # p1.rv_error, Time(rv_times[0], format='decimalyear'), rv_times[-1], IWA, OWA, dMag0
    # )
    rv_curve_path = Path(data_path, "rv_curve", str(base_path) + ".csv")
    post_path = Path(data_path, "post", str(base_path) + ".p")
    chains_path = Path(data_path, "chains", str(base_path) + ".csv")
    pop_path = Path(
        data_path,
        "planet_population",
        str(p1.base_path)
        + (
            f"_imgtimei{img_times[0].decimalyear:.0f}_imgtimef{img_times[-1].decimalyear:.0f}_imgtimestep{(img_times[1].decimalyear*u.yr-img_times[0].decimalyear*u.yr).to(u.d):.1f}_{pop_types[0]}".replace(
                " ", ""
            )
            + ".p"
        ),
    )
    chains = pd.read_csv(chains_path)
    pop = pd.read_pickle(pop_path)

    failure_mode, failure_rates, first_if_list, if_times, if_vals, df_times, df_vals = pap.failure_mode_calculations(data_path, p1, img_times, [IWA, OWA, dMag0], [], [], method_suffix=pop_types[0], return_failure_times=True)
    print(failure_mode)
    if failure_mode == 'both_failure':
        plot_if = True
        plot_df = True
    elif failure_mode == 'intermittent_failure':
        plot_if = True
        plot_df = False
    elif failure_mode == 'dispersion_failure':
        plot_if = False
        plot_df = True
    else:
        plot_if = False
        plot_df = False

    # ppop_options = {
        # "n_fits": 50000,
        # "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
        # "cov_samples": 1000,
        # "fixed_inc": None,
        # "fixed_f_sed": 3,
        # "fixed_p": 0.367,
        # "t0": Time(rv_times[0], format='decimalyear'),
    # }
    # pop = PlanetPopulation('SAG13', 10*u.pc, 1*u.M_sun, [0], {}, {}, options=ppop_options, EXOSIMS_dict=EXOSIMS_dict)
    # Fixing direction to be correct
    pcolor='red'
    # RV curve colormap
    # norm = mpl.colors.Normalize(vmin=min(p1_rv_curve['truevel']), vmax=max(p1_rv_curve['truevel']))
    # my_cmap = plt.get_cmap('coolwarm')
    IWA_s = np.tan(IWA.to(u.rad).value)*p1.dist_to_star.to(u.AU).value
    OWA_s = np.tan(OWA.to(u.rad).value)*p1.dist_to_star.to(u.AU).value
    dMag_range = np.linspace(16, 35, 5)
    pdets = []
    for fnum, current_time in enumerate( tqdm( img_times ) ):
        time_jd = Time(Time(current_time, format='decimalyear').jd, format='jd')

        planet_pos = p1.calc_position_vectors(time_jd)
        planet_alpha, planet_dMag = p1.prop_for_imaging(time_jd)

        pop_pos = pop.calc_position_vectors(time_jd)
        pop_alpha, pop_dMag = pop.prop_for_imaging(time_jd)

        fig, pvt_ax = plt.subplots(ncols=1, figsize=[15, 6])
        # Set the star up in the center
        # ax.scatter(0, 0, s=250, zorder=2, color='white')

        # # Add the planets at their current location
        # ax.scatter(planet_pos[0].to(u.AU), planet_pos[1].to(u.AU), s=20, label='Planet', zorder=3, edgecolor='black', color=pcolor)
        # ax.scatter(pop_pos[0, :].to(u.AU), pop_pos[1, :].to(u.AU), s=0.1, alpha=0.5, label=f'Constructed orbits ({pop_types[0]})', color='white')

        # # Now set plot limits
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_xlabel('x (AU)')
        # ax.set_ylabel('y (AU)')
        # ax.set_title(f'Ratio: {ratio:.2f}')
        # ax.legend(loc='upper right')

        # # Add coronagraph feature
        # IWA_patch = mpatches.Circle((0,0), IWA_s, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        # ax.add_patch(IWA_patch)
        # # OWA
        # inner_OWA_vertices = make_circle(OWA_s)
        # outer_OWA_vertices = make_circle(7)
        # vertices = np.concatenate((outer_OWA_vertices[::1], inner_OWA_vertices[::-1]))
        # codes = np.ones( len(inner_OWA_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        # codes[0] = mpath.Path.MOVETO
        # all_codes = np.concatenate((codes, codes))
        # path = mpath.Path(vertices, all_codes)
        # patch = mpatches.PathPatch(path, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        # ax.add_patch(patch)

        # # Add completeness
        IWA_ang = IWA
        OWA_ang = OWA
        meet_criteria = sum((planet_dMag < dMag0) & (OWA_ang.to(u.arcsec).value > planet_alpha.to(u.arcsec).value) & (planet_alpha.to(u.arcsec).value > IWA_ang.to(u.arcsec).value))
        pdet = int(meet_criteria)
        # # ax.annotate(f"{completeness:.2f}", xy=(0, 2.1), ha='center', va='center', zorder=6, size=20)

        # # Set up correct aspect ratio
        # ax.set_aspect('equal', 'box')

        # # Set up the alpha vs dMag
        # a_dMag_ax.set_xlabel(r'$\alpha$ (arcsec)')
        # a_dMag_ax.set_ylabel(r'$\Delta$mag')
        # a_dMag_ax.set_xlim([0, 0.125])
        # a_dMag_ax.set_ylim([dMag_range[0], dMag_range[-1]])

        # detectability_line = mpl.lines.Line2D([IWA_ang.to(u.arcsec).value, OWA_ang.to(u.arcsec).value], [dMag0, dMag0], color='red')
        # IWA_line = mpl.lines.Line2D([IWA_ang.to(u.arcsec).value, IWA_ang.to(u.arcsec).value], [dMag_range[0], dMag_range[-1]], color='red')
        # OWA_line = mpl.lines.Line2D([OWA_ang.to(u.arcsec).value, OWA_ang.to(u.arcsec).value], [dMag_range[0], dMag_range[-1]], color='red')
        # a_dMag_ax.add_line(detectability_line)
        # a_dMag_ax.add_line(IWA_line)
        # a_dMag_ax.add_line(OWA_line)
        # a_dMag_ax.scatter(planet_alpha.to(u.arcsec).value, planet_dMag, s=20, label='Planet', edgecolor='black', color=pcolor, zorder=3)
        # a_dMag_ax.scatter(pop_alpha.to(u.arcsec).value, pop_dMag, s=0.1, alpha=0.5, label=f'Constructed orbits {pop_types[0]}', color='white')
        # a_dMag_ax.legend(loc='upper right')
        # a_dMag_ax.set_title(f'Time: {current_time.decimalyear:.2f}')

        # pdet axes
        pdets.append(pdet)
        # planet_visibility = p1.planet_visibility
        pop_pdets = pop.percent_detectable[:fnum+1]
        cumulative_times = img_times[:fnum+1].value
        # pvt_ax.scatter(cumulative_times, pdets, c=pdets, cmap=pvt_cmap, norm=pvt_norm)
        pvt_ax.plot(cumulative_times, pdets, linestyle='--', label='Planet', color=pcolor)
        pvt_ax.plot(cumulative_times, pop_pdets, linestyle='-', label=f'Constructed orbits {pop_types[0]}', color='white')

        # Add failures to plot
        if plot_if:
            if_ids = np.where(if_times <= current_time.decimalyear)[0]
            if len(if_ids) is 0:
                pass
            else:
                plt_if_times = if_times[if_ids]
                plt_if_vals = np.array(if_vals)[if_ids]
                pvt_ax.scatter(plt_if_times, plt_if_vals, marker='v', label="Intermittent Failure", zorder=3)
        if plot_df:
            df_ids = np.where(df_times <= current_time.decimalyear)[0]
            if len(df_ids) is 0:
                pass
            else:
                plt_df_times = df_times[df_ids]
                plt_df_vals = np.array(df_vals)[df_ids]
                pvt_ax.scatter(plt_df_times, plt_df_vals, marker='d', label="DF", zorder=3)

        pvt_ax.set_xlim([img_times[0].value, img_times[-1].value])
        pvt_ax.set_ylim([-.05, 1.05])
        pvt_ax.set_ylabel(f'$P_{{det}}$')
        pvt_ax.set_xlabel('Year')
        pvt_ax.set_title(f'HabEx $P_{{det}}$: {pop.percent_detectable[fnum]:.3f}')
        pvt_ax.legend(loc='lower right')

        fig.tight_layout()
        # fig.savefig(Path(f'../figures/comp_constraint/frame-{fnum}.png'), dpi=150)
        fig.savefig(Path(f'../figures/{pop_types[0]}_if/if_only_ratio-{ratio:.2f}-frame-{fnum}.png'), dpi=150)
