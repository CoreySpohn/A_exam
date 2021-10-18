import numpy as np
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from planet import Planet
from pathlib import Path
import matplotlib as mpl

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
    # REVERT CHANGES to above
    p2_inputs = {'a': 2*u.AU,
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
    # Fixing direction to be correct
    font = {'size': 23}
    plt.rc("font", **font)
    plt.style.use("dark_background")
    # RV curve colormap
    norm = mpl.colors.Normalize(vmin=min(p1_rv_curve['truevel']), vmax=max(p1_rv_curve['truevel']))
    my_cmap = plt.get_cmap('coolwarm')
    betas = np.random.uniform(0, np.pi, 1000)*u.rad
    betas.sort()
    lams = p1.lambert_func(betas)
    # for beta in np.sin(np.random.uniform(0, np.pi, 1000)):
        # lam.append(p1.lambert_func(beta))
    fig, ax = plt.subplots(figsize=[10, 5])
    ax.plot(betas, lams)
    ax.set_xlabel(r'$\beta$ (rad)')
    ax.set_ylabel(r'$\phi(\beta)$')
    plt.show()
    fig.savefig('../figures/lambert/phi.png', dpi=150)

    # for fnum, current_time in enumerate( times ):
        # time_jd = Time(Time(current_time, format='decimalyear').jd, format='jd')
        # p1_pos = p1.calc_position_vectors(time_jd)
        # p2_pos = p2.calc_position_vectors(time_jd)

        # # Get the beta values for photometry calculations
        # p1_r = np.linalg.norm(p1_pos, axis=0)
        # p1_beta = np.arccos(p1_pos[2, :].value / p1_r.value) * u.rad
        # p2_r = np.linalg.norm(p2_pos, axis=0)
        # p2_beta = np.arccos(p2_pos[2, :].value / p2_r.value) * u.rad
        # p1_phase = p1.lambert_func(p1_beta)
        # p2_phase = p2.lambert_func(p2_beta)

        # fig, (p1_vis_ax, p1_rv_ax) = plt.subplots(nrows=1, ncols=2, figsize=[12, 4.5])
        # if p1_pos[2] > 0:
            # p1_order=1
        # else:
            # p1_order=3
        # # Set the star up in the center
        # p1_star_pos_offset = -0.05*np.array([p1_pos[0].to(u.AU), p1_pos[1].to(u.AU)])
        # p1_vis_ax.scatter(p1_star_pos_offset[0], p1_star_pos_offset[1], s=250, zorder=2, c=p1_rv_curve['truevel'][fnum], cmap=my_cmap, norm=norm)

        # # Add the planets at their current location
        # p1_vis_ax.scatter(p1_pos[0].to(u.AU), p1_pos[1].to(u.AU), s=10+(5*(p1_beta.value)), label=p1.planet_label, zorder=p1_order)

        # # Now set plot limits
        # p1_vis_ax.set_xlim([-2, 2])
        # p1_vis_ax.set_ylim([-2, 2])
        # p1_vis_ax.set_xlabel('x (AU)')
        # p1_vis_ax.set_ylabel('y (AU)')


        # # Set up correct aspect ratio
        # p1_vis_ax.set_aspect('equal', 'box')

        # # Set up the RV plot
        # current_p1_rvs = p1_rv_curve['truevel'][:fnum+1]
        # current_p1_times = p1_rv_curve['time'][:fnum+1]
        # current_p2_rvs = p2_rv_curve['truevel'][:fnum+1]
        # current_p2_times = p2_rv_curve['time'][:fnum+1]

        # p1_rv_ax.scatter(Time(list(current_p1_times), format='jd').decimalyear, current_p1_rvs, c=current_p1_rvs, cmap=my_cmap, norm=norm)
        # p1_rv_ax.set_xlim([times[0], times[-1]])
        # p1_rv_ax.set_ylim([-0.1, .1])
        # p1_rv_ax.set_xlabel('Year')
        # p1_rv_ax.set_ylabel('RV (m/s)')
        # p1_rv_ax.axhline(0, color='white', linestyle='--', alpha = 0.5)
        # if fnum+1 == len(times):
            # # On the last frame label the semi-amplitude
            # p1_rv_ax.annotate(r'RV semi-amplitude, $K$ (m/s)', xytext=(2000.5, -0.01), xy=(2000.5, max(p1_rv_curve['truevel'])), ha='center', va='center', arrowprops=dict(arrowstyle='->'), zorder=10)
        # fig.tight_layout()
        # p1_vis_ax.legend(loc='upper center')
        # fig.savefig(Path(f'../figures/rv_example/frame-{fnum}.png'), dpi=150)
        # # plt.show()
        # # breakpoint()

