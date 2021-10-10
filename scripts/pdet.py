import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from pathlib import Path
from matplotlib import cm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib as mpl

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))

if __name__ == '__main__':
    p1_inputs = {'a': 1*u.AU,
                 'e': 0,
                 'W': 0*u.rad,
                 'I': 90*u.degree,
                 'w': 0*u.rad,
                 'Mp': 1*u.M_earth,
                 'Rp': 1*u.R_earth,
                 'f_sed': 0,
                 'p': 0.367,
                 'M0': 0*u.rad,
                 't0': Time(2000, format='decimalyear'),
                 'rv_error': 0.001*u.m/u.s}
    p1 = Planet(r'1 $M_\oplus$', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p1_inputs)
    p2_inputs = {'a': 1*u.AU,
                 'e': 0,
                 'W': 0*u.rad,
                 'I': 5.73917048*u.degree,
                 'w': 0*u.rad,
                 'Mp': 10*u.M_earth,
                 'Rp': 1*u.R_earth,
                 'f_sed': 0,
                 'p': 0.367,
                 'M0': 0*u.rad,
                 't0': Time(2000, format='decimalyear'),
                 'rv_error': 0.001*u.m/u.s}
    p2 = Planet('10 $M_\oplus$', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p2_inputs)
    times = np.linspace(2000, 2001, 100)
    future_times = np.linspace(2001, 2002, 100)
    plt.style.use("dark_background")
    cc = plt.Circle((0, 0), 1)
    dMag_range = np.linspace(16, 35, 5)
    # Making custom colormap for unnormalized values
    dMag_norm = mpl.colors.Normalize(vmin=dMag_range[0], vmax=dMag_range[-1])
    # colors = ['white', 'black']
    # tuples = list(zip(map(dMag_norm, dMag_range), colors))
    # my_cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

    p1_rv_curve = p1.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    p2_rv_curve = p2.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    my_cmap = plt.get_cmap('binary')
    edge_cmap = plt.get_cmap('plasma')
    plt.set_cmap('binary')
    font = {'size': 13}
    plt.rc("font", **font)
    norm = mpl.colors.Normalize(vmin=min(p1_rv_curve['truevel']), vmax=max(p1_rv_curve['truevel']))
    rv_cmap = plt.get_cmap('coolwarm')
    p1_alphas = []
    p1_dMags = []
    p2_alphas = []
    p2_dMags = []
    pdets = []
    pvt_cmap = plt.get_cmap('RdYlGn')
    pvt_norm = mpl.colors.Normalize(vmin=0.25, vmax=1)
    for fnum, current_time in enumerate(future_times):
        time_jd = Time(Time(current_time, format='decimalyear').jd, format='jd')
        p1_pos = p1.calc_position_vectors(time_jd)
        p2_pos = p2.calc_position_vectors(time_jd)

        # Get the beta values for photometry calculations
        p1_r = np.linalg.norm(p1_pos, axis=0)
        p1_beta = np.arccos(p1_pos[2, :].value / p1_r.value) * u.rad
        p2_r = np.linalg.norm(p2_pos, axis=0)
        p2_beta = np.arccos(p2_pos[2, :].value / p2_r.value) * u.rad
        p1_phase = p1.lambert_func(p1_beta)
        p2_phase = p2.lambert_func(p2_beta)

        p1_edge_color = edge_cmap(0.25)
        p2_edge_color = edge_cmap(0.5)

        # Calculating dMag for photometry
        p1_alpha, p1_dMag = p1.prop_for_imaging(time_jd)
        p2_alpha, p2_dMag = p2.prop_for_imaging(time_jd)
        p1_s = (np.tan(p1_alpha)*p1.dist_to_star).to(u.AU).value
        p2_s = (np.tan(p2_alpha)*p2.dist_to_star).to(u.AU).value
        p1_FR = 10**(p1_dMag/(-2.5))
        p2_FR = 10**(p2_dMag/(-2.5))

        fig, (a_dMag_ax, pvt_ax) = plt.subplots(ncols=2, figsize=[16/1.5, 7/1.5])

        p1_color = my_cmap(dMag_norm(p1_dMag))
        p2_color = my_cmap(dMag_norm(p2_dMag))

        # Add coronagraph feature
        IWA = 0.5
        OWA = 1.15

        dMag0 = 26.5
        # visibility threshold
        p1_phot = (p1_dMag < dMag0) & (OWA > p1_s > IWA)
        p2_phot = (p2_dMag < dMag0) & (OWA > p2_s > IWA)
        pdet = (int(p1_phot[0])+int(p2_phot[0]))/2

        # Add the alpha vs dMag plot
        a_dMag_ax.set_xlabel(r'$\alpha$ (arcsec)')
        a_dMag_ax.set_ylabel(r'$\Delta$mag')
        a_dMag_ax.set_xlim([0, 0.125])
        a_dMag_ax.set_ylim([dMag_range[0], dMag_range[-1]])

        a_dMag_ax.set_title(f'$P_{{det}}({Time(time_jd, format="jd").decimalyear:.2f})$ = {(int(p1_phot[0])+int(p2_phot[0]))/2}')
        IWA_ang = np.arctan(IWA*u.AU/p1.dist_to_star).to(u.arcsec).value
        OWA_ang = np.arctan(OWA*u.AU/p1.dist_to_star).to(u.arcsec).value
        detectability_line = mpl.lines.Line2D([IWA_ang, OWA_ang], [dMag0, dMag0], color='red')
        a_dMag_ax.add_line(detectability_line)
        a_dMag_ax.scatter(p1_alpha.to(u.arcsec).value, p1_dMag, color=p1_color, edgecolor=p1_edge_color)
        a_dMag_ax.scatter(p2_alpha.to(u.arcsec).value, p2_dMag, color=p2_color, edgecolor=p2_edge_color)

        # Now make the pdet plot
        pdets.append(pdet)
        cumulative_times = future_times[:fnum+1]
        # pvt_ax.scatter(cumulative_times, pdets, c=pdets, cmap=pvt_cmap, norm=pvt_norm)
        pvt_ax.scatter(cumulative_times, pdets)
        pvt_ax.set_xlim([future_times[0], future_times[-1]])
        pvt_ax.set_ylim([-.05, 1.05])
        pvt_ax.set_ylabel(f'$P_{{det}}$')
        pvt_ax.set_xlabel('Year')

        # cbar.ax.plot([0, 0.1], [p2_phot-.1, p2_phot+0.1], color=p2_edge_color, linewidth=100)
        fig.tight_layout()
        fig.savefig(Path(f'../figures/pdet/frame-{fnum}.png'), dpi=150)

