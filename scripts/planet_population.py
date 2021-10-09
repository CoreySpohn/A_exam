from pathlib import Path

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from astropy.time import Time
from keplertools import fun
from planet import Planet
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import EXOSIMS.PlanetPopulation as PP
import EXOSIMS.MissionSim


class PlanetPopulation(Planet):
    def __init__(
        self,
        planet_label,
        dist_to_star,
        star_mass,
        star_fe,
        plot_kwargs,
        scatter_kwargs,
        options,
        chains=None,
        rv_inputs=None,
        base_planet=None,
        base_planet_errors=None,
        EXOSIMS_dict=None,
    ):
        # issue_chains = pd.read_csv(Path('data', 'chains', 'a2.00AU_e0.00_W0.00rad_I2.50rad_w0.00rad_Mp26.56earthMass_Rp4.48earthRad_Ms1.00solMass_error0.0100_ti2000.00_tf2010.00_IWA0.058arcsec_OWA10.0arcsec_dMag026.5.csv'))
        # if chains == issue_chains:
        self.planet_label = planet_label
        self.dist_to_star = dist_to_star
        self.Ms = star_mass
        self.star_fe = star_fe
        self.plot_kwargs = plot_kwargs
        self.scatter_kwargs = scatter_kwargs

        # load up all the options
        self.cov_samples = options["cov_samples"]
        self.n_fits = options["n_fits"]
        self.droppable_cols = options["droppable_cols"]
        self.fixed_inc = options["fixed_inc"]
        self.fixed_f_sed = options["fixed_f_sed"]
        self.fixed_p = options["fixed_p"]
        self.num = self.n_fits

        if chains is not None:
            # Store raw chains
            # self.chains = chains

            # Get the highest probability chains for the covariance matrix
            self.samples_for_cov = (
                chains.pipe(self.start_pipeline)
                .pipe(self.sort_by_lnprob)
                .pipe(self.get_samples_for_covariance)
                .pipe(self.drop_columns)
            )

            # Get covariance of the whole dataframe
            # self.samples_for_cov = chains.pipe(self.start_pipeline).pipe(self.drop_columns)

            # Calculate the covariance matrix
            self.cov_df = self.samples_for_cov.cov()
            self.chains_means = (
                chains.pipe(self.start_pipeline).pipe(self.drop_columns).mean()
            )
            chain_samples_np = np.random.multivariate_normal(
                self.chains_means, self.cov_df, size=self.n_fits
            )
            chain_samples = pd.DataFrame(chain_samples_np, columns=self.cov_df.keys())

            # Use those samples and assign the values
            self.T = chain_samples.per1.to_numpy() * u.d
            self.secosw = chain_samples.secosw1.to_numpy()
            self.sesinw = chain_samples.sesinw1.to_numpy()
            self.K = chain_samples.k1.to_numpy()
            self.T_c = Time(chain_samples.tc1.to_numpy(), format="jd")

            # Now use the method from Planet to solve for keplerian parameters
            self.rv_to_kep()
        elif base_planet is not None:
            # So in this case we need to sample over the inclination space and within the error
            # bars if they are given
            self.base_planet = base_planet
            if base_planet_errors is None:
                self.T = np.ones(self.num) * base_planet.T
                self.secosw = np.ones(self.num) * base_planet.secosw
                self.sesinw = np.ones(self.num) * base_planet.sesinw
                self.K = np.ones(self.num) * base_planet.K
                self.T_c = Time(np.ones(self.num) * base_planet.T_c.jd, format="jd")
            else:
                self.std = base_planet_errors
                # now with the errors we have to do rv_to_kep
                self.T = (
                    np.random.normal(
                        base_planet.T.value, self.std["per1"], size=self.num
                    )
                    * base_planet.T.unit
                )
                self.secosw = np.random.normal(
                    base_planet.secosw, self.std["secosw1"], size=self.num
                )
                self.sesinw = np.random.normal(
                    base_planet.sesinw, self.std["sesinw1"], size=self.num
                )
                self.K = (
                    np.random.normal(base_planet.K.value, self.std["k1"], size=self.num)
                    * base_planet.K.unit
                )
                self.T_c = Time(
                    np.random.normal(
                        base_planet.T_c.jd, self.std["tc1"], size=self.num
                    ),
                    format="jd",
                )
            self.rv_to_kep()
        elif EXOSIMS_dict is not None:
            script=EXOSIMS_dict['script']
            sim = EXOSIMS.MissionSim.MissionSim(script)
            self.mu = sim.PlanetPopulation.mu
            self.a, self.e, self.p, self.Rp = sim.PlanetPopulation.gen_plan_params(self.num)
            I, W, w = sim.PlanetPopulation.gen_angles(self.num)
            self.I = I.to(u.rad)
            self.w = w.to(u.rad)
            self.W = W.to(u.rad)
            self.M0 = np.random.uniform(0, 2*np.pi, int(self.num))*u.rad # mean anomaly
            self.t0 = options['t0']
        # Assign some intermediate values
        self.orbElem = (self.a, self.e, self.W, self.I, self.w)
        self.n = (np.sqrt(self.mu / self.a ** 3)).decompose()

    def get_samples_for_covariance(self, df):
        return df.head(self.cov_samples)

    def sort_by_lnprob(self, df):
        # return df.sort_values("lnprobability", ascending=False)
        return df.sort_values("lnprobability", ascending=False)

    def drop_columns(self, df):
        return df.drop(columns=self.droppable_cols)

    def drop_nan_rows(self, df):
        return df.dropna()

    def start_pipeline(self, df):
        return df.copy()

    def get_head(self, df):
        return df.head(1)

    def del_duplicates(self, df):
        return df.drop_duplicates()

    def kde_obj(self, x):
        return -self.kde.evaluate(x)

    def get_kde_estimate(self, chains):
        # Have to drop the unnecessary columns and transpose to use the gaussian_kde function
        self.kde_input = (
            chains.pipe(self.start_pipeline)
            .pipe(self.drop_columns)
            .pipe(self.drop_nan_rows)
            .T
        )
        try:
            kde = gaussian_kde(self.kde_input)
        except Exception as err:
            print(f"gaussian_kde error: {err}")
            # breakpoint()
            print(f"nans: {np.count_nonzero(np.isnan(self.kde_input))}")
            print(f"infs: { np.count_nonzero(np.isinf(self.kde_input)) }")

        # Store it
        self.kde = kde

        # Find the mean values from the population to use as the initial vector for minimize
        self.kde_x0 = chains.pipe(self.start_pipeline).pipe(self.drop_columns).mean()
        # Now to find the set of parameters that returns the maximum evaluation
        # from the kernel density estimate
        self.kde_min_results = minimize(self.kde_obj, self.kde_x0.values)
        return self.kde_min_results
