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
                 'rv_error': 0.05*u.m/u.s}
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
                 'rv_error': 0.05*u.m/u.s}
    p2 = Planet('10 $M_\oplus$', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p2_inputs)
    times = np.linspace(2000, 2001, 100)
    plt.style.use("dark_background")
    cc = plt.Circle((0, 0), 1)
    dMag_range = np.linspace(16, 35, 5)
    # Making custom colormap for unnormalized values
    dMag_norm = mpl.colors.Normalize(vmin=dMag_range[0], vmax=dMag_range[-1])
    # colors = ['white', 'black']
    # tuples = list(zip(map(dMag_norm, dMag_range), colors))
    # my_cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)

    p1_rv_curve = p1.simulate_rv_observations(Time(times, format='decimalyear'), p1.rv_error)
    p2_rv_curve = p2.simulate_rv_observations(Time(times, format='decimalyear'), p2.rv_error)

    future_times = np.linspace(2001, 2002, 100)
    p1_rv_curve_future = p1.simulate_rv_observations(Time(future_times, format='decimalyear'), 0.001*u.m/u.s)

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
    for fnum, current_time in enumerate( future_times ):
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

        fig, (p1_vis_ax, p2_vis_ax) = plt.subplots(nrows=2, figsize=[22.4/1.5, 9/1.5])


        if p1_pos[2] > 0:
            p1_order=1
        else:
            p1_order=3
        # Set the star up in the center
        p1_star_pos_offset = -0.05*np.array([p1_pos[0].to(u.AU), p1_pos[1].to(u.AU)])
        p1_vis_ax.scatter(p1_star_pos_offset[0], p1_star_pos_offset[1], s=250, zorder=2, c=p1_rv_curve_future['truevel'][fnum], cmap=rv_cmap, norm=norm)
        p2_star_pos_offset = -0.05*np.array([p2_pos[0].to(u.AU), p2_pos[1].to(u.AU)])
        p2_vis_ax.scatter(p2_star_pos_offset[0], p2_star_pos_offset[1], s=250, zorder=2, c=p1_rv_curve_future['truevel'][fnum], cmap=rv_cmap, norm=norm)

        p1_color = my_cmap(dMag_norm(p1_dMag))
        p2_color = my_cmap(dMag_norm(p2_dMag))

        # Add the planets at their current location
        p1_vis_ax.scatter(p1_pos[0].to(u.AU), p1_pos[1].to(u.AU), s=10+(5*(p1_beta.value)), label=p1.planet_label, zorder=p1_order, color=p1_color, edgecolor=p1_edge_color)
        p2_vis_ax.scatter(p2_pos[0].to(u.AU), p2_pos[1].to(u.AU), s=100+(5*(p2_beta.value)), label=p2.planet_label, zorder=1, color=p2_color, edgecolor=p2_edge_color)

        # Now set plot limits
        p1_vis_ax.set_xlim([-2, 2])
        p1_vis_ax.set_ylim([-2, 2])
        # p1_vis_ax.set_xlabel('AU')
        p1_vis_ax.set_ylabel('AU')

        p2_vis_ax.set_xlim([-2, 2])
        p2_vis_ax.set_ylim([-2, 2])
        p2_vis_ax.set_xlabel('AU')
        p2_vis_ax.set_ylabel('AU')

        # Set up correct aspect ratio
        p1_vis_ax.set_aspect('equal', 'box')
        p2_vis_ax.set_aspect('equal', 'box')

        # Add coronagraph feature
        IWA = 0.5
        OWA = 1.15
        IWA_patch = mpatches.Circle((0,0), IWA, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p1_vis_ax.add_patch(IWA_patch)
        IWA_patch = mpatches.Circle((0,0), IWA, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p2_vis_ax.add_patch(IWA_patch)
        inner_OWA_vertices = make_circle(OWA)
        outer_OWA_vertices = make_circle(3)
        vertices = np.concatenate((outer_OWA_vertices[::1], inner_OWA_vertices[::-1]))
        codes = np.ones( len(inner_OWA_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        all_codes = np.concatenate((codes, codes))
        path = mpath.Path(vertices, all_codes)
        patch = mpatches.PathPatch(path, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p1_vis_ax.add_patch(patch)
        patch = mpatches.PathPatch(path, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p2_vis_ax.add_patch(patch)

        # if fnum == 0:
            # p1_vis_ax.annotate(f'Outer working angle', (0, 1.3), va='top', ha='center', zorder=6)
            # p1_vis_ax.annotate(f'Inner\nworking\nangle', (0, 0.49), va='top', ha='center', zorder=6)
            # p2_vis_ax.annotate(f'Outer working angle', (0, 1.3), va='top', ha='center', zorder=6)
            # p2_vis_ax.annotate(f'Inner\nworking\nangle', (0, 0.49), va='top', ha='center', zorder=6)
        # OWA_vertices = make_circle(1)
        # p1_vis_ax.scatter(0, 0, s=1000, facecolors='black', edgecolor='grey', zorder=5, alpha=0.9)
        # p2_vis_ax.scatter(0, 0, s=1000, facecolors='black', edgecolor='grey', zorder=5, alpha=0.9)

        # p1_vis_ax.scatter(0, 0, s=1499, facecolors='none', edgecolor='grey', zorder=5, alpha=0.9)
        # p1_vis_ax.scatter(0, 0, s=1500, facecolors='none', edgecolor='black', linewidth=100, zorder=5, alpha=0.9)
        # p2_vis_ax.scatter(0, 0, s=1500, facecolors='none', edgecolor='black', linewidth=100, zorder=5, alpha=0.9)

        # Weird stuff to make colormap work
        sm = plt.cm.ScalarMappable(cmap=my_cmap)
        sm._A=[]
        sm.set_array(dMag_range)
        # breakpoint()
        fig.subplots_adjust(left=0.0, right=0.4)
        cbar_ax = fig.add_axes([0.515, 0.1, 0.02, 0.8])
        cbar = fig.colorbar(sm, cax=cbar_ax, label=r'$\Delta$mag')
        # Add bar on colorbar to indicate current brightness
        cbar.ax.axhline(p1_dMag, color=p1_edge_color, linewidth=2, markeredgecolor='white')
        cbar.ax.axhline(p2_dMag, color=p2_edge_color, linewidth=2, markeredgecolor='white')
        dMag0 = 26.5
        threshold_colors = { True: 'green', False: 'red'}
        cbar.ax.axhline(dMag0, color='red', linewidth=2,
                        markeredgecolor='white', label='dMag0')
        # visibility threshold
        p1_phot = (p1_dMag < dMag0) & (OWA > p1_s > IWA)
        p2_phot = (p2_dMag < dMag0) & (OWA > p2_s > IWA)
        for spine in p1_vis_ax.spines.values():
            spine.set_edgecolor(threshold_colors[p1_phot[0]])
        for spine in p2_vis_ax.spines.values():
            spine.set_edgecolor(threshold_colors[p2_phot[0]])

        # Add the alpha vs dMag plot
        a_dMag_ax = fig.add_axes([0.65, 0.1, 0.3, 0.8])
        a_dMag_ax.set_xlabel(r'$\alpha$ (arcsec)')
        a_dMag_ax.set_ylabel(r'$\Delta$mag')
        a_dMag_ax.set_xlim([0, 0.125])
        a_dMag_ax.set_ylim([dMag_range[0], dMag_range[-1]])

        IWA_ang = np.arctan(IWA*u.AU/p1.dist_to_star).to(u.arcsec).value
        OWA_ang = np.arctan(OWA*u.AU/p1.dist_to_star).to(u.arcsec).value
        detectability_line = mpl.lines.Line2D([IWA_ang, OWA_ang], [dMag0, dMag0], color='red')
        a_dMag_ax.add_line(detectability_line)
        a_dMag_ax.set_title(f'$P_{{det}}({Time(time_jd, format="jd").decimalyear:.2f})$ = {(int(p1_phot[0])+int(p2_phot[0]))/2}')
        # p1_alphas.append(p1_alpha.to(u.arcsec).value)
        # p1_dMags.append(p1_dMag)
        # p2_alphas.append(p2_alpha.to(u.arcsec).value)
        # p2_dMags.append(p2_dMag)
        # a_dMag_ax.scatter(p1_alphas, p1_dMags, c=p1_dMags, cmap=my_cmap, norm=dMag_norm, edgecolor=p1_edge_color)
        # a_dMag_ax.scatter(p2_alphas, p2_dMags, c=p2_dMags, cmap=my_cmap, norm=dMag_norm, edgecolor=p2_edge_color)
        a_dMag_ax.scatter(p1_alpha.to(u.arcsec).value, p1_dMag, color=p1_color, edgecolor=p1_edge_color)
        a_dMag_ax.scatter(p2_alpha.to(u.arcsec).value, p2_dMag, color=p2_color, edgecolor=p2_edge_color)

        # cbar.ax.plot([0, 0.1], [p2_phot-.1, p2_phot+0.1], color=p2_edge_color, linewidth=100)
        # fig.tight_layout()
        p1_vis_ax.set_title('Fitted orbits')
        p1_vis_ax.legend(loc='upper center')
        p2_vis_ax.legend(loc='upper center')
        # fig.savefig(Path(f'../figures/mass_inclination_comparision/frame-{fnum:04}.png'))

        # Set up the RV plot
        fig.subplots_adjust(left=0.2, right=0.65)
        rv_ax = fig.add_axes([0.07, 0.1, 0.23, 0.8])

        current_p1_rvs = p1_rv_curve_future['truevel'][:fnum+1]
        current_p1_times = Time(list(p1_rv_curve_future['time'][:fnum+1]), format='jd').decimalyear
        # p1_rvs = p1_rv_curve['truevel']
        initial_rvs = p1_rv_curve['vel']
        initial_times = Time(list(p1_rv_curve['time']), format='jd').decimalyear

        rv_ax.errorbar(initial_times, initial_rvs, yerr=p1.rv_error.decompose().value, alpha=0.5, fmt='o', zorder= 1)
        rv_ax.scatter(initial_times, initial_rvs, c=initial_rvs, cmap=rv_cmap, norm=norm)
        rv_ax.scatter(current_p1_times, current_p1_rvs, c=current_p1_rvs, cmap=rv_cmap, norm=norm)
        # rv_ax.scatter(current_p1_times, current_p1_rvs, )
        rv_ax.set_xlim([times[0], future_times[-1]])
        rv_ax.set_ylim([-0.1, .1])
        rv_ax.set_xlabel('Year')
        rv_ax.set_ylabel('RV (m/s)')
        rv_ax.axhline(0, color='white', linestyle='--', alpha = 0.5)
        rv_ax.axvline(2001, color='white', linestyle='--', alpha = 0.5)
        rv_ax.annotate('RV\nobservations', xy=(2000.5, -0.09), ha='center', va='center')
        if fnum > 0:
            rv_ax.annotate('Fitted\nRV', xy=(2001.5, -0.09), ha='center', va='center')
        if fnum == 0:
            # Hide the other subplots on the first frame
            p1_vis_ax.set_visible(False)
            p2_vis_ax.set_visible(False)
            cbar_ax.set_visible(False)
            a_dMag_ax.set_visible(False)
        if fnum == 1:
            # Hide the other subplots on the first frame
            a_dMag_ax.set_visible(False)


        fig.savefig(Path(f'../figures/rv_pdet_2/frame-{fnum}.png'), dpi=150)

