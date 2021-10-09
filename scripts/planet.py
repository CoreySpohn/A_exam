import warnings
from pathlib import Path

import astropy
import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
import radvel.orbit as rvo
import radvel.utils as rvu
from astropy.time import Time
from keplertools import fun as kt


class Planet:
    def __init__(
        self,
        planet_label,
        dist_to_star,
        star_mass,
        star_fe,
        plot_kwargs,
        scatter_kwargs,
        rv_inputs=None,
        keplerian_inputs=None,
    ):

        # Assign the base values
        self.planet_label = planet_label
        self.dist_to_star = dist_to_star
        self.Ms = star_mass
        self.star_fe = star_fe
        self.num = 1
        self.plot_kwargs = plot_kwargs
        self.scatter_kwargs = scatter_kwargs

        # Calculate the luminosity of the star, assuming main-sequence
        if self.Ms < 2 * u.M_sun:
            self.Ls = const.L_sun * (self.Ms / const.M_sun) ** 4
        else:
            self.Ls = 1.4 * const.L_sun * (self.Ms / const.M_sun) ** 3.5

        assert (rv_inputs is not None) or (
            keplerian_inputs is not None
        ), f"No inputs given for {planet_label}"
        assert (rv_inputs is None) or (
            keplerian_inputs is None
        ), f"Two input sets given for {planet_label}"
        if (rv_inputs is not None) and (keplerian_inputs is None):
            # When the inputs are given as rv values the keplerian orbital parameters should be calculated as well
            # RV inputs should be given as a dictionary
            self.rv_inputs = rv_inputs
            self.T = rv_inputs["period"]
            self.secosw = rv_inputs["secosw"]
            self.sesinw = rv_inputs["sesinw"]
            self.K = rv_inputs["K"]
            self.T_c = rv_inputs["T_c"]
            self.fixed_inc = rv_inputs["fixed_inc"]
            self.fixed_f_sed = rv_inputs["fixed_f_sed"]
            self.fixed_p = rv_inputs["fixed_p"]

            # This calculates the keplerian parameters for the planet
            self.rv_to_kep()
        elif (keplerian_inputs is not None) and (rv_inputs is None):
            # In this case we can just assign the values
            self.a = keplerian_inputs["a"].decompose()
            self.e = keplerian_inputs["e"]
            self.W = keplerian_inputs["W"]
            self.I = keplerian_inputs["I"]
            self.w = keplerian_inputs["w"]
            self.Mp = keplerian_inputs["Mp"]
            self.Rp = keplerian_inputs["Rp"]
            self.f_sed = keplerian_inputs["f_sed"]
            self.p = keplerian_inputs["p"]
            self.M0 = keplerian_inputs["M0"]
            self.t0 = keplerian_inputs["t0"]
            self.rv_error = keplerian_inputs["rv_error"]

            # Now calcluate the other values
            self.mu = (const.G * (self.Mp + self.Ms)).decompose()
            self.T = (2 * np.pi * np.sqrt(self.a ** 3 / self.mu)).to(u.d)
            self.w_p = self.w
            self.w_s = (self.w + np.pi * u.rad) % (2 * np.pi * u.rad)
            self.secosw = np.sqrt(self.e) * np.cos(self.w)
            self.sesinw = np.sqrt(self.e) * np.sin(self.w)

            # Because we have the mean anomaly at an epoch we can calculate the
            # time of periastron as t0 - T_e where T_e is the time since periastron
            # passage
            T_e = (self.T * self.M0 / (2 * np.pi * u.rad)).decompose()
            self.T_p = self.t0 - T_e

            # Calculate the time of conjunction
            self.T_c = Time(
                rvo.timeperi_to_timetrans(
                    self.T_p.jd, self.T.value, self.e, self.w_s.value
                ),
                format="jd",
            )

            # Find the semi-amplitude
            self.K = (
                (2 * np.pi * const.G / self.T) ** (1 / 3.0)
                * (self.Mp * np.sin(self.I) / self.Ms ** (2 / 3.0))
                * (1 - self.e ** 2) ** (-1 / 2)
            ).decompose()

        self.orbElem = (self.a, self.e, self.W, self.I, self.w)
        self.n = (np.sqrt(self.mu / self.a ** 3)).decompose()

        # Classify the planet into Kopparapu bins
        # self.classify_planet()

    def rv_to_kep(self):
        """
        Assumes that secosw and sesinw are the argument of periapsis of the star not planet

        Sets the class attributes:
            e: eccentricity
            w: argument of periapsis
            I: inclination
            Mp: planet_mass
            W: Longitude of the ascending node
            mu: gravitational parameter
            a: semi-major axis
            w_s: star's argument of periapsis
            w: planet's argument of periapsis
            M0: planet's mean anomaly at epoch (time of conjunction)
            t0: time of epoch (time of conjunction)
            orbElem: tuple of (a, e, W, I, w_p)
        """
        secosw = self.secosw
        sesinw = self.sesinw
        T = self.T
        K = self.K
        T_c = self.T_c
        e = secosw ** 2 + sesinw ** 2
        if isinstance(e, np.ndarray):
            e[np.where((e > 1) | (e == 0))] = 0.0001
        else:
            if (e > 1) or (e == 0):
                e = 0.0001
        self.e = e
        # Mass of planet and inclination
        self.Msini = rvu.Msini(K, T, np.ones(T.size) * self.Ms, e, Msini_units="earth")
        # print('1')
        if self.fixed_inc is None:
            # Without a fixed inclination the sinusoidal inclination distribution is sampled from
            # and used to calcluate a corresponding mass
            Icrit = np.arcsin(
                (self.Msini * u.M_earth).value
                / ((0.0800 * u.M_sun).to(u.M_earth)).value
            )
            Irange = [Icrit, np.pi - Icrit]
            C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
            I = np.arccos(np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=len(e)))
            self.I = I * u.rad
        else:
            # When a fixed inclination is given we don't need to sample anything
            I = self.fixed_inc
            self.I = I
        # print('3')
        # Planet mass from inclination
        self.Mp = np.abs(self.Msini / np.sin(I)) * u.M_earth
        # print('4')
        # if round(self.Mp, 2) == 26.56*u.M_earth:
        # print('YAAS')
        # print('5')
        # Use modified FORCASTER model from plandb
        self.Rp = self.RfromM(self.Mp)
        # print('6')
        # Set geometric albedo
        if self.fixed_p is not None:
            self.p = self.fixed_p
        # print('7')
        # Set a random value for longitude of the ascending node, could also
        # just not track it
        self.W = np.random.uniform(0, 180) * u.deg
        # print('8')

        # Now with inclination, planet mass, and longitude of the ascending node
        # we can calculate the remaining parameters
        self.mu = (const.G * (self.Mp + self.Ms)).decompose()
        self.a = ((self.mu * (T / (2 * np.pi)) ** 2) ** (1 / 3)).decompose()
        # print('9')

        # Finding the planet's arguement of periapsis
        self.w_s = np.arctan2(sesinw, secosw) * u.rad
        self.w = (self.w_s + np.pi * u.rad) % (2 * np.pi * u.rad)

        # Finding the mean anomaly at time of conjunction
        nu_p = (np.pi / 2 * u.rad - self.w_s) % (2 * np.pi * u.rad)
        E_p = 2 * np.arctan2(np.sqrt((1 - e)) * np.tan(nu_p / 2), np.sqrt((1 + e)))
        self.M0 = (E_p - e * np.sin(E_p) * u.rad) % (2 * np.pi * u.rad)

        # set initial epoch to the time of conjunction since that's what M0 is
        self.t0 = Time(T_c, format="jd")
        self.T_p = rvo.timetrans_to_timeperi(T_c, self.T, e, self.w.value)

        # Find the semi-amplitude
        self.K = (
            (2 * np.pi * const.G / self.T) ** (1 / 3.0)
            * (self.Mp * np.sin(self.I) / self.Ms ** (2 / 3.0))
            * (1 - self.e ** 2) ** (-1 / 2)
        ).decompose()

        if self.fixed_f_sed is None:
            self.f_sed = np.random.choice(
                [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 6],
                self.num,
                p=[0.099, 0.001, 0.005, 0.01, 0.025, 0.28, 0.3, 0.28],
            )
        else:
            self.f_sed = np.ones(self.num) * self.fixed_f_sed

    def RfromM(self, m):
        """
        Given masses m (in Earth masses) return radii (in Earth radii) \
        based on modified forecaster
        """
        if type(m) == u.quantity.Quantity:
            m = m.to(u.M_earth).value
        m = np.array(m, ndmin=1)
        R = np.zeros(m.shape)

        S = np.array([0.2790, 0, 0, 0, 0.881])
        C = np.array([np.log10(1.008), 0, 0, 0, 0])
        T = np.array(
            [
                2.04,
                95.16,
                (u.M_jupiter).to(u.M_earth),
                ((0.0800 * u.M_sun).to(u.M_earth)).value,
            ]
        )
        Rj = u.R_jupiter.to(u.R_earth)
        Rs = 8.522  # saturn radius

        S[1] = (np.log10(Rs) - (C[0] + np.log10(T[0]) * S[0])) / (
            np.log10(T[1]) - np.log10(T[0])
        )
        C[1] = np.log10(Rs) - np.log10(T[1]) * S[1]

        S[2] = (np.log10(Rj) - np.log10(Rs)) / (np.log10(T[2]) - np.log10(T[1]))
        C[2] = np.log10(Rj) - np.log10(T[2]) * S[2]

        C[3] = np.log10(Rj)

        C[4] = np.log10(Rj) - np.log10(T[3]) * S[4]

        inds = np.digitize(m, np.hstack((0, T, np.inf)))
        # print('RFROMM NOWWW\n\n')
        # print(f'm: {m}')
        # print(f'inds = {inds}')
        # print(f'length of inds = {len(inds)}')
        # print(f'R = {R}')
        # print(f'length of R = {len(R)}')
        for j in range(1, inds.max() + 1):
            # print(f'j = {j}')
            # try:
            R[inds == j] = 10.0 ** (C[j - 1] + np.log10(m[inds == j]) * S[j - 1])
            # except Exception as e:
            # print(e)

        return R * u.R_earth

    def mean_anom(self, t):
        # t and t0 are Time quantities so that they can easily be manipulated
        # n is in standard SI units so the times are converted to seconds
        M1 = self.n * (t - self.t0).to(u.s) * u.rad
        M = (M1 + self.M0) % (2 * np.pi * u.rad)
        return M

    def calc_position_vectors(self, t):
        # This will find the radial and velocity vectors at an epoch
        M = self.mean_anom(t)
        E = kt.eccanom(M.value, self.e)
        a, e, O, I, w = self.a.decompose(), self.e, self.W, self.I, self.w
        A = np.vstack(
            (
                a * (np.cos(O) * np.cos(w) - np.sin(O) * np.cos(I) * np.sin(w)),
                a * (np.sin(O) * np.cos(w) + np.cos(O) * np.cos(I) * np.sin(w)),
                a * np.sin(I) * np.sin(w),
            )
        )

        B = np.vstack(
            (
                -a
                * np.sqrt(1 - e ** 2)
                * (np.cos(O) * np.sin(w) + np.sin(O) * np.cos(I) * np.cos(w)),
                a
                * np.sqrt(1 - e ** 2)
                * (-np.sin(O) * np.sin(w) + np.cos(O) * np.cos(I) * np.cos(w)),
                a * np.sqrt(1 - e ** 2) * np.sin(I) * np.cos(w),
            )
        )
        if np.isscalar(self.mu) and not (np.isscalar(E)):
            rpv = np.matmul(A, np.array((np.cos(E) - e), ndmin=2)) + np.matmul(
                B, np.array(np.sin(E), ndmin=2)
            )
        else:
            rpv = np.matmul(A, np.diag(np.cos(E) - e)) + np.matmul(
                B, np.diag(np.sin(E))
            )
        return rpv

    def calc_p_phi(self, beta, photdict, bandinfo):
        photinterps2 = photdict["photinterps"]
        feinterp = photdict["feinterp"]
        distinterp = photdict["distinterp"]

        (l, band, bw, ws, wstep) = bandinfo
        lum = (self.Ms.to(u.M_sun).value) ** 3.5
        if np.isnan(lum):
            lum_fix = 1
        else:
            lum_fix = (10 ** lum) ** 0.5  # Since lum is log base 10 of solar luminosity

        p_phi = np.zeros(len(beta))
        for i, f_sed_val in enumerate(np.unique(self.f_sed)):
            # fe = self.fe_vals[i]
            fe = self.star_fe
            tmpinds = self.f_sed == f_sed_val
            betatmp = beta[tmpinds]
            binds = np.argsort(betatmp)
            p_phi[tmpinds] = (
                photinterps2[float(feinterp(fe))][
                    float(distinterp(np.mean(self.dist_to_star) / lum_fix))
                ][f_sed_val](betatmp.to(u.deg).value[binds], ws).sum(1)
                * wstep
                / bw
            )[np.argsort(binds)].flatten()

        p_phi[np.isinf(p_phi)] = np.nan
        p_phi[p_phi <= 0.0] = 1e-16
        return p_phi

    def lambert_func(self, beta):
        return (np.sin(beta) * u.rad + (np.pi * u.rad - beta) * np.cos(beta)) / (
            np.pi * u.rad
        )

    def prob_det_pc(self, times, IWA, OWA, dMag0):
        a, e, I, w, Rp = (
            self.a.to(u.m).value,
            self.e,
            self.I.to(u.rad).value,
            self.w.to(u.rad).value,
            self.Rp.to(u.m).value,
        )
        n, t0, M0, dist_to_star, p = (
            self.n.to(1 / u.s).value,
            self.t0.jd,
            self.M0.to(u.rad).value,
            self.dist_to_star.to(u.m).value,
            self.p,
        )
        return prob_det_c_wrapper(
            times.jd,
            a,
            e,
            I,
            w,
            Rp,
            n,
            t0,
            M0,
            dist_to_star,
            p,
            IWA.to(u.rad).value,
            OWA.to(u.rad).value,
            dMag0,
            self.num,
        )

    def prob_det(self, times, IWA, OWA, dMag0):
        a, e, I, w, Rp = (
            self.a.to(u.m).value,
            self.e,
            self.I.to(u.rad).value,
            self.w.to(u.rad).value,
            self.Rp.to(u.m).value,
        )
        n, t0, M0, dist_to_star, p = (
            self.n.to(1 / u.s).value,
            self.t0.jd,
            self.M0.to(u.rad).value,
            self.dist_to_star.to(u.m).value,
            self.p,
        )

        return c_prob_det(
            times.jd,
            a,
            e,
            I,
            w,
            Rp,
            n,
            t0,
            M0,
            dist_to_star,
            p,
            IWA.to(u.rad).value,
            OWA.to(u.rad).value,
            dMag0,
            self.num,
        )

    def prop_for_imaging_c(self, t):
        a, e, I, w, Rp = (
            self.a.to(u.m).value,
            self.e,
            self.I.to(u.rad).value,
            self.w.to(u.rad).value,
            self.Rp.to(u.m).value,
        )
        n, t0, M0, dist_to_star, p = (
            self.n.to(1 / u.s).value,
            self.t0.jd,
            self.M0.to(u.rad).value,
            self.dist_to_star.to(u.m).value,
            self.p,
        )

        WA, dMag = c_prop(t.jd, a, e, I, w, Rp, n, t0, M0, dist_to_star, p)
        return WA * u.rad, dMag

    def prop_for_imaging(self, t):
        # Calculates the working angle and deltaMag
        a, e, I, w = self.a, self.e, self.I, self.w
        M = self.mean_anom(t)
        E = kt.eccanom(M.value, self.e)
        nu = kt.trueanom(E, e) * u.rad
        r = a * (1 - e ** 2) / (1 + e * np.cos(nu))

        theta = nu + w
        rps = self.calc_position_vectors(t)
        s = (r / 4) * np.sqrt(
            4 * np.cos(2 * I)
            + 4 * np.cos(2 * theta)
            - 2 * np.cos(2 * I - 2 * theta)
            - 2 * np.cos(2 * I + 2 * theta)
            + 12
        )
        # Working with positive z away from observer
        beta = np.arccos(rps[2, :].value/r.decompose().value)*u.rad
        # For gas giants
        # p_phi = self.calc_p_phi(beta, photdict, bandinfo)
        # For terrestrial planets
        phi = self.lambert_func(beta)
        p_phi = self.p * phi

        # s = np.linalg.norm(rps[0:2, :], axis=0)
        # WA = (s / self.dist_to_star).decompose() * u.rad
        WA = np.arctan(s / self.dist_to_star).decompose()
        dMag = -2.5 * np.log10(p_phi * ((self.Rp / r).decompose()) ** 2).value
        return WA, dMag

    def prop_for_imaging_old(self, t):
        # Calculates the working angle and deltaMag
        rps = self.calc_position_vectors(t)
        r = np.linalg.norm(rps, axis=0)
        beta = np.arccos(rps[2, :].value / r.value) * u.rad
        # For gas giants
        # p_phi = self.calc_p_phi(beta, photdict, bandinfo)
        # For terrestrial planets
        phi = self.lambert_func(beta)
        p_phi = self.p * phi

        s = np.linalg.norm(rps[0:2, :], axis=0)
        WA = (s / self.dist_to_star).decompose() * u.rad

        dMag = -2.5 * np.log10(p_phi * ((self.Rp / r).decompose()) ** 2).value
        return WA, dMag, beta

    def calc_vs(self, t, return_nu=False):
        M = self.mean_anom(t)
        E = kt.eccanom(M, self.e)
        nu = kt.trueanom(E, self.e) * u.rad
        vs = (
            np.sqrt(const.G / ((self.Mp + self.Ms) * self.a * (1 - self.e ** 2)))
            * self.Mp
            * np.sin(self.I)
            * (np.cos(self.w_s + nu) + self.e * np.cos(self.w_s))
        )
        if return_nu:
            return vs.decompose(), nu
        else:
            return vs.decompose()

    def simulate_rv_observations(self, times, error):
        rv_error = error.decompose().value
        # Save the rv observation times
        self.rv_observation_times = Time(times, format="jd")
        # Create a dataframe to hold the planet data
        column_names = ["time", "truevel", "tel", "svalue", "time_year"]

        rv_data_df = pd.DataFrame(
            0, index=np.arange(len(times)), columns=column_names, dtype=object
        )
        # nu_array = np.zeros([len(times), 2])
        nu_array = np.zeros(len(times))

        # Loop through the times and calculate the radial velocity at the desired time
        for i, t in enumerate(self.rv_observation_times):
            # M = planet.mean_anom(t)
            # E = kt.eccanom(M.value, planet.e)
            # nu = kt.trueanom(E, planet.e) * u.rad
            vs, nu = self.calc_vs(t, return_nu=True)
            rv_data_df.at[i, "time"] = t.jd  # Time of observation in julian days
            rv_data_df.at[i, "time_year"] = t.decimalyear
            # Velocity at observation in m/s
            rv_data_df.at[i, "truevel"] = float(vs.decompose().value)
            # This is saying it's all the same inst
            rv_data_df.at[i, "tel"] = "i"

            # appending to nu array
            nu_array[i] = nu[0].to(u.rad).value
        # Calculate a random velocity offset or error based on the rv uncertainty
        vel_offset = np.random.normal(scale=rv_error, size=len(times))
        vel_offset_df = pd.DataFrame({"err_offset": vel_offset})
        rv_data_df = pd.concat([rv_data_df, vel_offset_df], axis=1)  # Append

        # This is simply an array of the one sigma error with some noise added
        errvel = np.ones(len(times)) * rv_error + np.random.normal(
            scale=rv_error / 10, size=len(times)
        )
        errvel_df = pd.DataFrame({"errvel": errvel})
        rv_data_df = pd.concat([rv_data_df, errvel_df], axis=1)

        # Add the errors onto the velocities
        adjusted_vels = rv_data_df["truevel"] + vel_offset
        vel_df = pd.DataFrame({"vel": adjusted_vels})
        rv_data_df = pd.concat([rv_data_df, vel_df], axis=1)

        # Add true anomaly
        nu_df = pd.DataFrame({"nu": nu_array})
        rv_data_df = pd.concat([rv_data_df, nu_df], axis=1)

        return rv_data_df

    def gen_base_path(self, error, initial_time, final_time, IWA, OWA, dMag0):
        self.base_path = Path(
            f"a{self.a.to(u.AU):.2f}_e{self.e:.2f}_W{self.W:.2f}_I{self.I:.2f}_w{self.w:.2f}_Mp{self.Mp.to(u.M_earth):.2f}_Rp{self.Rp.to(u.R_earth):.2f}_Ms{self.Ms.to(u.M_sun):.2f}_error{error.value:.4f}_ti{initial_time.value:.2f}_tf{final_time.value:.2f}_IWA{IWA}_OWA{OWA}_dMag0{dMag0}".replace(
                " ", ""
            )
        )
        return self.base_path

    def classify_planet(self):
        """
        This determines the Kopparapu bin of the planet
        This is adapted from the EXOSIMS SubtypeCompleteness method classifyPlanets so that EXOSIMS isn't a mandatory import
        """
        Rp = self.Rp.to("earthRad").value
        a = self.a.to("AU").value
        e = self.e

        # Find the stellar flux at the planet's location as a fraction of earth's
        earth_Lp = const.L_sun / (1 * (1 + (0.0167 ** 2) / 2)) ** 2
        self.Lp = (
            self.Ls / (self.a.to("AU").value * (1 + (self.e ** 2) / 2)) ** 2 / earth_Lp
        )

        # Find Planet Rp range
        Rp_bins = np.array([0, 0.5, 1.0, 1.75, 3.5, 6.0, 14.3, 11.2 * 4.6])
        Rp_lo = Rp_bins[:-1]
        Rp_hi = Rp_bins[1:]
        Rp_types = [
            "Sub-Rocky",
            "Rocky",
            "Super-Earth",
            "Sub-Neptune",
            "Sub-Jovian",
            "Jovian",
            "Super-Jovian",
        ]
        self.L_bins = np.array(
            [
                [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
                [1000, 182, 1.0, 0.28, 0.0035, 5e-5],
                [1000, 187, 1.12, 0.30, 0.0030, 5e-5],
                [1000, 188, 1.15, 0.32, 0.0030, 5e-5],
                [1000, 220, 1.65, 0.45, 0.0030, 5e-5],
                [1000, 220, 1.65, 0.40, 0.0025, 5e-5],
                [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
                [1000, 220, 1.68, 0.45, 0.0025, 5e-5],
            ]
        )

        # Find the bin of the radius
        self.Rp_bin = np.digitize(Rp, Rp_bins) - 1
        try:
            self.Rp_type = Rp_types[self.Rp_bin]
        except:
            print(f"Error handling Rp_type of planet with Rp_bin of {self.Rp_bin}")
            self.Rp_type = None

        # TODO Fix this to give correct when at edge cases since technically they're not straight lines

        # index of planet temp. cold,warm,hot
        L_types = ["Hot", "Warm", "Cold"]
        specific_L_bins = self.L_bins[self.Rp_bin, :]
        self.L_bin = np.digitize(self.Lp.decompose().value, specific_L_bins) - 1
        try:
            self.L_type = L_types[self.L_bin]
        except:
            print(f"Error handling L_type of planet with L_bin of {self.L_bin}")

        # Now assign the colors that will get used when plotting
        self.subtype_color = ["red", "yellow", "blue", "black", "green"][self.L_bin]
        self.subtype_marker = [".", "X", "P", "v", "s", "D", "H", "<"][self.Rp_bin]
