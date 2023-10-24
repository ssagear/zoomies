import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import corner
import pickle
from astropy.io import ascii
from tqdm import tqdm
import pandas as pd

from pyia import GaiaData
from astropy.table import Table, join

import jax
import jax.numpy as jnp
from jax_cosmo.scipy.integrate import simps as simpson
from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as spline

import numpyro
from numpyro import distributions as dist, infer
numpyro.set_host_device_count(4)

import arviz as az

from src.quick_jz import calc_jz


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def find_nearest_index(array, value):
    """Gets the index of the nearest element of array to a value."""
    array = np.asarray(array)
    idx = np.nanargmin((np.abs(array - value)))
    return idx



class KinematicAgeSpline:

    def __init__(self, lnJz, age, age_err):

        self.lnJz = lnJz
        self.age = age
        self.age_err = age_err


    def monotonic_quadratic_spline(self, x, y, x_eval):
        """
        The zeroth element in the knot value array is the value of the spline at x[0], but
        all other values passed in via y are the *derivatives* of the function at the knot
        locations x[1:].
        """

        # Checked that using .at[].set() is faster than making padded arrays and stacking
        x = jnp.array(x)
        y = jnp.array(y)
        x_eval = jnp.array(x_eval)

        N = 3 * (len(x) - 1)
        A = jnp.zeros((N, N))
        b = jnp.zeros((N,))
        A = A.at[0, :3].set([x[0] ** 2, x[0], 1])
        b = b.at[0].set(y[0])
        A = A.at[1, :3].set([2 * x[1], 1, 0])
        b = b.at[1].set(y[1])

        for i, n in enumerate(2 * jnp.arange(1, len(x) - 1, 1), start=1):
            A = A.at[n, 3 * i : 3 * i + 3].set([2 * x[i], 1, 0])
            b = b.at[n].set(y[i])
            A = A.at[n + 1, 3 * i : 3 * i + 3].set([2 * x[i + 1], 1, 0])
            b = b.at[n + 1].set(y[i + 1])

        for j, m in enumerate(jnp.arange(2 * (len(x) - 1), N - 1)):
            A = A.at[m, 3 * j : 3 * j + 3].set([x[j + 1] ** 2, x[j + 1], 1])
            A = A.at[m, 3 * (j + 1) : 3 * (j + 1) + 3].set([-x[j + 1] ** 2, -x[j + 1], -1])

        A = A.at[-1, 0].set(1.0)

        coeffs = jnp.linalg.solve(A, b)

        # Determine the interval that x lies in
        ind = jnp.digitize(x_eval, x) - 1
        ind = 3 * jnp.clip(ind, 0, len(x) - 2)
        coeff_ind = jnp.stack((ind, ind + 1, ind + 2), axis=0)

        xxx = jnp.stack([x_eval**2, x_eval, jnp.ones_like(x_eval)], axis=0)
        f = jnp.sum(coeffs[coeff_ind] * xxx, axis=0)

        return f

    def truncated_density_quadratic_model(self, age, age_err, lnJz, age_knots, ln_dens_knots):
        """Monotonic spline with truncated likelihood, scatter and Poisson density"""

        import numpyro
        from jax_cosmo.scipy.integrate import simps as simpson
        from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline as spline

        # lnJz vs age spline
        K_knots_age = len(age_knots)
        age_knot_vals = numpyro.sample("age_knot_vals", dist.Uniform(jnp.concatenate((jnp.array([-4.]), jnp.full(K_knots_age-1, 0.))), jnp.full(K_knots_age, 5.)))
        #age_knot_vals = numpyro.sample("age_knot_vals", dist.Uniform(jnp.concatenate((jnp.array([-4.]), jnp.full(K_knots_age-1, 0.))), jnp.full(K_knots_age, 8.)))

        # intrinsic scatter
        #lnV = numpyro.sample("lnV", dist.Normal(1, 1))
        lnV = numpyro.sample("lnV", dist.Normal(9,5))
        V = numpyro.deterministic("V", jnp.exp(lnV))

        # density spline
        K_knots_ln_dens = len(ln_dens_knots)
        ln_dens_knot_vals = numpyro.sample("dens_knot_vals", dist.Uniform(jnp.full(K_knots_ln_dens, -10), jnp.full(K_knots_ln_dens, 15)))

        ln_dens_func = spline(ln_dens_knots, ln_dens_knot_vals)
        ln_rate = jnp.sum(ln_dens_func(age))
        V_eff = simpson(lambda x: jnp.exp(ln_dens_func(x)), 0.0, 15, N=256)
        numpyro.factor("poisson", ln_rate - V_eff)

        # put a deterministic in the numpyro model -- 
        # optional argument passed into - ln Jz-age grid. if defined, evaluate # density

        with numpyro.plate("data", len(age)):

            true_age = numpyro.sample("true_age", dist.TruncatedNormal(age, age_err, low=0, high=14), obs=age)
            numpyro.sample("lnJz_pred", dist.Normal(self.monotonic_quadratic_spline(age_knots, age_knot_vals, true_age), jnp.sqrt(V)), obs=lnJz)



    def fit_mono_spline(self, ln_dens_knots=jnp.linspace(-1, 15, 15), age_knots=jnp.linspace(-1, 14, 5),\
                        num_warmup=1000, num_samples=1000, num_chains=2, progress_bar=True):
        
        import arviz as az
        
        self.ln_dens_knots = ln_dens_knots
        self.age_knots = age_knots
        
        dens_sampler = infer.MCMC(
            infer.NUTS(self.truncated_density_quadratic_model),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        print('Fitting line...')
        dens_sampler.run(
            jax.random.PRNGKey(0),
            age=self.age,
            age_err=self.age_err,
            lnJz=self.lnJz,
            ln_dens_knots=self.ln_dens_knots,
            age_knots=self.age_knots
        );

        self.dens_sampler  = dens_sampler
        
        inf_data = az.from_numpyro(dens_sampler)
        self.inf_data = inf_data


    def evaluate_spline(self, eval_grid=np.linspace(0, 14, 1000), k=0):
        """Evaluate the spline for a given posterior index (k) over a grid"""

        self.grid = eval_grid
        self.eval_spline = self.monotonic_quadratic_spline(self.age_knots, self.dens_sampler.get_samples()['age_knot_vals'][k], eval_grid)


    def plot_fit(self, Jz_age_bins=(np.linspace(-4, 6, 32), np.linspace(0, 14, 32)), eval_grid=np.linspace(0, 14, 1000)):

        if not hasattr(self, 'eval_spline'):
            self.evaluate_spline(eval_grid=eval_grid, k=0)

        plt.hist2d(self.lnJz, self.age, bins=Jz_age_bins, cmap='Blues')
        plt.plot(self.eval_spline, self.grid, color='red', label='Spline fit', lw=2)

        plt.ylabel('Age (Gyr)', fontsize=15);
        plt.xlabel('$ln(J_z)$', fontsize=15);
        plt.colorbar(label='# density');
        plt.legend()
        plt.title('Age vs. $ln(J_z)$', fontsize=15)

        plt.show()


    def evaluate_ages(self, lnJz_sample, eval_grid=np.linspace(0, 14, 1000), k=0):
        """Evaluate the age posterior for a given posterior index (k) and given lnJz over a grid"""

        from scipy.stats import norm
        from scipy.integrate import simpson as scipy_simpson
        from tqdm import tqdm

        if not hasattr(self, 'eval_spline'):
            self.evaluate_spline(eval_grid=eval_grid, k=0)

        self.age_knot_vals = self.dens_sampler.get_samples()['age_knot_vals'][k]
        self.V_samp = np.exp(self.dens_sampler.get_samples()['lnV'][k])
        

        eval_pdf = []
        i16 = []
        i50 = []
        i84 = []
        a_arr = []

        if isinstance(lnJz_sample, float) or isinstance(lnJz_sample, int):

            logP = norm.logpdf(lnJz_sample, loc=self.eval_spline, scale=np.sqrt(self.V_samp))
            P = np.exp(logP)
            P /= scipy_simpson(x=eval_grid, y=P)

            eval_pdf = P

            i16 = np.abs(np.cumsum(np.diff(eval_grid)[0] * P) - 0.16).argmin()
            i50 = np.abs(np.cumsum(np.diff(eval_grid)[0] * P) - 0.50).argmin()
            i84 = np.abs(np.cumsum(np.diff(eval_grid)[0] * P) - 0.84).argmin()

        else:

            for n in tqdm(range(len(lnJz_sample))):
                logP = norm.logpdf(lnJz_sample[n], loc=self.eval_spline, scale=np.sqrt(self.V_samp))
                P = np.exp(logP)
                P /= scipy_simpson(x=self.grid, y=P)

                eval_pdf.append(P)

                i16.append(np.abs(np.cumsum(np.diff(self.grid)[0] * P) - 0.16).argmin())
                i50.append(np.abs(np.cumsum(np.diff(self.grid)[0] * P) - 0.50).argmin())
                i84.append(np.abs(np.cumsum(np.diff(self.grid)[0] * P) - 0.84).argmin())

        self.eval_grid = eval_grid
        self.eval_pdf = np.array(eval_pdf)

        return eval_grid, np.array(eval_pdf), np.array(i16), np.array(i50), np.array(i84)
        

    def get_mode_and_percentiles(self, eval_pdf, eval_grid=np.linspace(0, 14, 1000)):

        from tqdm import tqdm

        

        full_results = []

        if eval_pdf.ndim > 1:

            for star in tqdm(range(len(eval_pdf))):

                mode = eval_grid[np.argmax(eval_pdf[star])]
                inc_trials = np.arange(0, 14, 0.1)
                star_results = []

                for inc in inc_trials:

                    leftlim = mode-inc
                    rightlim = mode+inc

                    if leftlim <= 0:
                        leftlim = 0

                    if rightlim >= 14:
                        rightlim=14

                    integral = np.trapz(self.eval_pdf[star], x=np.linspace(leftlim, rightlim, len(eval_pdf[star])))
                    star_results.append([inc, leftlim, rightlim, integral, mode])

                full_results.append(np.array(star_results))

        elif eval_pdf.ndim == 1:

            mode = eval_grid[np.argmax(eval_pdf)]
            inc_trials = np.arange(0, 14, 0.01)

            for inc in inc_trials:

                leftlim = mode-inc
                rightlim = mode+inc

                if leftlim <= 0:
                    leftlim = 0

                if rightlim >= 14:
                    rightlim=14

                integral = np.trapz(self.eval_pdf, x=np.linspace(leftlim, rightlim, len(eval_pdf)))
                full_results.append([inc, leftlim, rightlim, integral, mode])


        full_results = np.array(full_results)
        return full_results


    def get_sigmas(self, full_results):

        full_best_sigmas = []

        if full_results.ndim > 2:

            for i in range(len(full_results)):

                results = full_results[i]
                best_sigma = results[find_nearest_index(results[:,3], 0.68)]
                full_best_sigmas.append(best_sigma)

        elif full_results.ndim == 2:

            results = full_results
            best_sigma = results[find_nearest_index(results[:,3], 0.68)]
            full_best_sigmas.append(best_sigma)

        full_best_sigmas = np.array(full_best_sigmas)
        return full_best_sigmas

    def get_mode_sigma(self, eval_pdf):

        full_results = self.get_mode_and_percentiles(eval_pdf)
        full_best_sigmas = self.get_sigmas(full_results)

        modes = full_best_sigmas[:,-1]
        sigmas = full_best_sigmas[:,0]
        lerr = full_best_sigmas[:,1]
        uerr = full_best_sigmas[:,2]

        if len(modes) == 1 and len(sigmas) == 1:
            return modes[0], sigmas[0], lerr[0], uerr[0]
        
        return modes, sigmas, lerr, uerr
        


