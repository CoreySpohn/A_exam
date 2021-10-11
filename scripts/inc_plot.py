import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats.kde import gaussian_kde

N = 1000000
# n = N/10
plt.style.use('dark_background')
# sin_curve = np.arccos(1 - 2 * np.random.uniform(size=N))
font = {'size': 16}
plt.rc("font", **font)
i_vals = np.linspace(0.37, 179.63, 10000)
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(i_vals, np.sin(i_vals*u.deg.to(u.rad))/2)
ax.set_xlabel(r'Inclination, $i$ (deg)')
ax.set_xticks(np.arange(0, 200, 20))
ax.set_ylabel('Density')
ax.set_title('Inclination prior')
ax.axvline(0.37, color='gray', linestyle='-', alpha=1, linewidth=1)
ax.axvline(179.63, color='gray', linestyle='-', alpha=1, linewidth=1)
ax.annotate(r'$i_{crit}$', xy=(-8, 0.25), rotation = 90, size=20)
ax.annotate(r'$i_{crit}$', xy=(171, 0.25), rotation = 90, size=20)
fig.savefig('../figures/inc_distribution.png', dpi=150)
# plt.show()
