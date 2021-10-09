import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from pathlib import Path
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
    p1_rv_curve = p1.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    p2_rv_curve = p2.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    plt.style.use("dark_background")
    font = {'size': 13}
    plt.rc("font", **font)
    cc = plt.Circle((0, 0), 1)
    norm = mpl.colors.Normalize(vmin=min(p1_rv_curve['truevel']), vmax=max(p1_rv_curve['truevel']))
    rv_cmap = plt.get_cmap('coolwarm')
    for fnum, current_time in enumerate( times ):
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

        fig, (p1_vis_ax, p2_vis_ax) = plt.subplots(ncols=2, figsize=[16/1.5, 9/1.5])
        if p1_pos[2] > 0:
            p1_order=1
        else:
            p1_order=3
        # Set the star up in the center
        p1_star_pos_offset = -0.05*np.array([p1_pos[0].to(u.AU), p1_pos[1].to(u.AU)])
        p1_vis_ax.scatter(p1_star_pos_offset[0], p1_star_pos_offset[1], s=250, zorder=2, c=p1_rv_curve['vel'][fnum], cmap=rv_cmap, norm=norm)
        p2_star_pos_offset = -0.05*np.array([p2_pos[0].to(u.AU), p2_pos[1].to(u.AU)])
        p2_vis_ax.scatter(p2_star_pos_offset[0], p2_star_pos_offset[1], s=250, zorder=2, c=p2_rv_curve['vel'][fnum], cmap=rv_cmap, norm=norm)

        # Add the planets at their current location
        p1_vis_ax.scatter(p1_pos[0].to(u.AU), p1_pos[1].to(u.AU), s=10+(5*(p1_beta.value)), label=p1.planet_label, zorder=p1_order)
        p2_vis_ax.scatter(p2_pos[0].to(u.AU), p2_pos[1].to(u.AU), s=100+(5*(p2_beta.value)), label=p2.planet_label, zorder=1)

        # Now set plot limits
        p1_vis_ax.set_xlim([-2, 2])
        p1_vis_ax.set_ylim([-2, 2])
        p1_vis_ax.set_xlabel('AU')
        p1_vis_ax.set_ylabel('AU')

        p2_vis_ax.set_xlim([-2, 2])
        p2_vis_ax.set_ylim([-2, 2])
        p2_vis_ax.set_xlabel('AU')
        # p2_vis_ax.set_ylabel('AU')

        # Set up correct aspect ratio
        p1_vis_ax.set_aspect('equal', 'box')
        p2_vis_ax.set_aspect('equal', 'box')

        # Add coronagraph feature
        IWA_patch = mpatches.Circle((0,0), 0.5, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p1_vis_ax.add_patch(IWA_patch)
        IWA_patch = mpatches.Circle((0,0), 0.5, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        p2_vis_ax.add_patch(IWA_patch)
        inner_OWA_vertices = make_circle(1.15)
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

        # On the first frame label the IWA and OWA
        if fnum == 0:
            # p1_vis_ax.annotate(f'Outer working angle', (0, -1.3), va='top', ha='center', zorder=6)
            # p1_vis_ax.annotate(f'Inner\nworking\nangle', (0, 0.49), va='top', ha='center', zorder=6)
            # p2_vis_ax.annotate(f'Outer working angle', (0, -1.3), va='top', ha='center', zorder=6)
            # p2_vis_ax.annotate(f'Inner\nworking\nangle', (0, 0.49), va='top', ha='center', zorder=6)
            p1_vis_ax.annotate("IWA (rad)", xy=(0, 0), xytext=(0, 0.6), ha='center', va='center', arrowprops=dict(arrowstyle='<-'), zorder= 10)
            p2_vis_ax.annotate("OWA (rad)", xy=(0, 0), xytext=(0, 1.25), ha='center', va='center', arrowprops=dict(arrowstyle='<-'), zorder= 10)
        # On the second frame label the separation of the planet
        if fnum == 1:
            s = (np.linalg.norm(p1_pos[0:2], axis=0)).to(u.AU).value
            p1_vis_ax.annotate(r"$s$ (AU)", xy=(0, 0), xytext=(s+0.33, 0), ha='center', va='center', arrowprops=dict(arrowstyle='<-'), zorder= 10)
        if fnum == 2:
            alpha, _ = p1.prop_for_imaging(time_jd)
            s = (np.tan(alpha)*p1.dist_to_star).to(u.AU).value
            p1_vis_ax.annotate(r"$\alpha$ (rad)", xy=(0, 0), xytext=(s+0.36, 0), ha='center', va='center', arrowprops=dict(arrowstyle='<-'), zorder= 10)
            # p2_vis_ax.annotate(r"$\alpha$ (radians)", xy=(0, 0), xytext=(0,-1.25), ha='center', va='center', arrowprops=dict(arrowstyle='<-'))
        fig.tight_layout()
        p1_vis_ax.legend(loc='upper center')
        p2_vis_ax.legend(loc='upper center')
        # fig.savefig(Path(f'../figures/mass_inclination_comparision/frame-{fnum:04}.png'))
        fig.savefig(Path(f'../figures/geometric_detection_criteria/frame-{fnum}.png'), dpi=150)

