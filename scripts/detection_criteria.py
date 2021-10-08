import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from pathlib import Path

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
    times = np.linspace(2000, 2004, 100)
    p1_rv_curve = p1.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    p2_rv_curve = p2.simulate_rv_observations(Time(times, format='decimalyear'), 0.001*u.m/u.s)
    breakpoint()
    plt.style.use("dark_background")
    for fnum, current_time in enumerate( times ):
        time_jd = Time(Time(current_time, format='decimalyear').jd, format='jd')
        p1_pos = p1.calc_position_vectors(time_jd)
        p2_pos = p2.calc_position_vectors(time_jd)
        fig, ((p1_vis_ax, p1_rv_ax), (p2_vis_ax, p2_rv_ax)) = plt.subplots(nrows=2, ncols=2, figsize=[9, 5])
        if p1_pos[2] > 0:
            p1_order=1
        else:
            p1_order=3
        # Set the star up in the center
        p1_vis_ax.scatter(0, 0, s=1000, zorder=2, color='yellow')
        p2_vis_ax.scatter(0, 0, s=1000, zorder=2, color='yellow')

        # Add the planets at their current location
        p1_vis_ax.scatter(p1_pos[0].to(u.AU), p1_pos[1].to(u.AU), label=p1.planet_label, zorder=p1_order)
        p2_vis_ax.scatter(p2_pos[0].to(u.AU), p2_pos[1].to(u.AU), s=100, label=p2.planet_label, zorder=1)

        # Now set plot limits
        p1_vis_ax.set_xlim([-2, 2])
        p1_vis_ax.set_ylim([-2, 2])
        p1_vis_ax.set_xlabel('AU')
        p1_vis_ax.set_ylabel('AU')

        p2_vis_ax.set_xlim([-2, 2])
        p2_vis_ax.set_ylim([-2, 2])
        p2_vis_ax.set_xlabel('AU')
        p2_vis_ax.set_ylabel('AU')

        # Set up correct aspect ratio
        p1_vis_ax.set_aspect('equal', 'box')
        p2_vis_ax.set_aspect('equal', 'box')

        # Set up the RV plot
        current_p1_rvs = p1_rv_curve['truevel'][:fnum+1]
        current_p1_times = p1_rv_curve['time'][:fnum+1]
        current_p2_rvs = p2_rv_curve['truevel'][:fnum+1]
        current_p2_times = p2_rv_curve['time'][:fnum+1]

        p1_rv_ax.plot(Time(list(current_p1_times), format='jd').decimalyear, current_p1_rvs)
        p2_rv_ax.plot(Time(list(current_p2_times), format='jd').decimalyear, current_p2_rvs)
        p1_rv_ax.set_xlim([times[0], times[-1]])
        p1_rv_ax.set_ylim([-0.1, .1])
        p2_rv_ax.set_xlim([times[0], times[-1]])
        p2_rv_ax.set_ylim([-0.1, .1])
        p1_rv_ax.set_xlabel('Year')
        p1_rv_ax.set_ylabel('RV (m/s)')
        p2_rv_ax.set_xlabel('Year')
        p2_rv_ax.set_ylabel('RV (m/s)')
        fig.tight_layout()
        fig.legend(loc='center left')
        # fig.savefig(Path(f'../figures/mass_inclination_comparision/frame-{fnum:04}.png'))
        fig.savefig(Path(f'../figures/mass_inclination_comparision/frame-{fnum}.png'))
        # plt.show()
        # breakpoint()

