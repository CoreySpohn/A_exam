import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from planet_population import PlanetPopulation
from pathlib import Path
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.patches as mpatches

def make_circle(r):
    t = np.arange(0, np.pi * 2.0, 0.01)
    t = t.reshape((len(t), 1))
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.hstack((x, y))

if __name__ == '__main__':
    times = np.linspace(2000, 2001, 100)
    EXOSIMS_dict = {'script': 'sag13.json'}
    ppop_options = {
        "n_fits": 50000,
        "droppable_cols": ["lnprobability", "curv", "gamma", "jit", "dvdt"],
        "cov_samples": 1000,
        "fixed_inc": None,
        "fixed_f_sed": 3,
        "fixed_p": 0.367,
        "t0": Time(times[0], format='decimalyear'),
    }

    sag13_pop = PlanetPopulation('SAG13', 10*u.pc, 1*u.M_sun, [0], {}, {}, options=ppop_options, EXOSIMS_dict=EXOSIMS_dict)
    # Fixing direction to be correct
    font = {'size': 13}
    plt.rc("font", **font)
    plt.style.use("dark_background")
    # RV curve colormap
    # norm = mpl.colors.Normalize(vmin=min(p1_rv_curve['truevel']), vmax=max(p1_rv_curve['truevel']))
    # my_cmap = plt.get_cmap('coolwarm')
    for fnum, current_time in enumerate( times ):
        time_jd = Time(Time(current_time, format='decimalyear').jd, format='jd')
        pop_pos = sag13_pop.calc_position_vectors(time_jd)
        pop_alpha, pop_dMag = sag13_pop.prop_for_imaging(time_jd)

        fig, ax = plt.subplots(figsize=[6, 6])
        # Set the star up in the center
        ax.scatter(0, 0, s=250, zorder=2, color='white')

        # Add the planets at their current location
        ax.scatter(pop_pos[0, :].to(u.AU), pop_pos[1, :].to(u.AU), s=0.1, alpha=0.5, label='SAG13')

        # Now set plot limits
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_xlabel('x (AU)')
        ax.set_ylabel('y (AU)')

        # IWA
        IWA = 0.5
        OWA = 1.15
        dMag0 = 26.5

        # Add coronagraph feature
        IWA_patch = mpatches.Circle((0,0), IWA, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        ax.add_patch(IWA_patch)
        # OWA
        inner_OWA_vertices = make_circle(OWA)
        outer_OWA_vertices = make_circle(3)
        vertices = np.concatenate((outer_OWA_vertices[::1], inner_OWA_vertices[::-1]))
        codes = np.ones( len(inner_OWA_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
        codes[0] = mpath.Path.MOVETO
        all_codes = np.concatenate((codes, codes))
        path = mpath.Path(vertices, all_codes)
        patch = mpatches.PathPatch(path, facecolor='grey', edgecolor='white', alpha=0.5, zorder=5)
        ax.add_patch(patch)

        # Add completeness
        IWA_ang = np.arctan(IWA*u.AU/sag13_pop.dist_to_star).to(u.arcsec).value
        OWA_ang = np.arctan(OWA*u.AU/sag13_pop.dist_to_star).to(u.arcsec).value
        meet_criteria = sum((pop_dMag < dMag0) & (OWA_ang > pop_alpha.to(u.arcsec).value) & (pop_alpha.to(u.arcsec).value > IWA_ang))
        completeness = meet_criteria/sag13_pop.num
        ax.set_title(f'Completeness: {completeness:.3f}')
        # ax.annotate(f"{completeness:.2f}", xy=(0, 2.1), ha='center', va='center', zorder=6, size=20)

        # Set up correct aspect ratio
        ax.set_aspect('equal', 'box')

        fig.tight_layout()
        fig.savefig(Path(f'../figures/completeness/frame-{fnum}.png'), dpi=150)
