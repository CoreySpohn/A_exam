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
                 'I': 100*u.degree,
                 'w': 0*u.rad,
                 'Mp': 1*u.M_earth,
                 'Rp': 1*u.R_earth,
                 'f_sed': 0,
                 'p': 0.367,
                 'M0': 0*u.rad,
                 't0': Time(2000, format='decimalyear'),
                 'rv_error': 0.001}
    p1 = Planet('1 Earth masses', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p1_inputs)
    p2_inputs = {'a': 1*u.AU,
                 'e': 0,
                 'W': 0*u.rad,
                 'I': 5.65169347*u.degree,
                 'w': 0*u.rad,
                 'Mp': 10*u.M_earth,
                 'Rp': 1*u.R_earth,
                 'f_sed': 0,
                 'p': 0.367,
                 'M0': 0*u.rad,
                 't0': Time(2000, format='decimalyear'),
                 'rv_error': 0.001}
    p2 = Planet('10 Earth masses', 10*u.pc, 1*u.M_sun, 0, {}, {}, keplerian_inputs=p2_inputs)
    times = np.linspace(2000, 2010, 100)
    for fnum, current_time in enumerate( times ):
        p1_pos = p1.calc_position_vectors(Time(current_time, format='jd'))
        p2_pos = p2.calc_position_vectors(Time(current_time, format='jd'))
        fig, ax = plt.subplots()
        ax.scatter(0, 0, s=10)
        ax.scatter(p1_pos[0].to(u.AU), p1_pos[2].to(u.AU), label=p1.planet_label)
        ax.scatter(p2_pos[0].to(u.AU), p2_pos[2].to(u.AU), s=100, label=p2.planet_label)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        fig.legend()
        fig.savefig(Path(f'../figures/mass_inclination_comparision/frame{fnum:04}.png'))
        # plt.show()
        # breakpoint()

