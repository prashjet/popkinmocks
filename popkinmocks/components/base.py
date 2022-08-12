import numpy as np
from scipy import stats, special
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm



class component(ABC):
    """Abstract base class to rerepsent a galaxy component

    A component is specified by it's joint density p(t,x,v,z) over stellar age
    t, 2D position x, line-of-sight velocity v, and metallicity z. Sub-classes
    of `component` correspond to specific (i) factorisations of the joint
    density and, (ii) implementations of the factors. Some details of p(t,x,v,z)
    are shared between all components:
    - the star formation history is independent of other variables - i.e. one
    can factor p(t,x,v,z) = p(t) ...,
    - p(t) is a beta distribution,
    - age-metallicity relations from chemical evolution model from eqns 3-10
    of Zhu et al 2020, parameterised by a depletion timescale `t_dep`,
    - spatial properties are set in co-ordinate system defined by a center and
    rotation relative to datacube x-axis.

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.

    """

    def __init__(self,
                 cube=None,
                 center=(0,0),
                 rotation=0.):
        self.cube = cube
        self.center = center
        self.rotation = rotation
        costh = np.cos(rotation)
        sinth = np.sin(rotation)
        rot_matrix = np.array([[costh, sinth],[-sinth,costh]])
        xxyy = np.dstack((self.cube.xx-self.center[0],
                          self.cube.yy-self.center[1]))
        xxyy_prime = np.einsum('kl,ijk',
                               rot_matrix,
                               xxyy,
                               optimize=True)
        self.xxp = xxyy_prime[:,:,0]
        self.yyp = xxyy_prime[:,:,1]
        self.get_z_interpolation_grid()

    def get_beta_a_b_from_lmd_phi(self, lmd, phi):
        """Convert from (total, mean) to (a,b) parameters of beta distribution

        Args:
            lmd: beta distribution total parameter, lmd>0
            phi: beta distribution mean parameter, 0<phi<1

        Returns:
            (a,b): shape parameters

        """
        a = lmd * phi
        b = lmd * (1. - phi)
        return a, b

    def set_p_t(self,
                lmd=None,
                phi=None,
                cdf_start_end=(0.05, 0.95)):
        """Set the star formation history

        p(t) = Beta(t; lmd, phi), where (lmd, phi) are (total, mean) parameters.
        Additionally this sets `self.t_pars` which is used for interpolating
        quantities against t. Any time varying quantity (e.g. disk size) is
        varied between start and end times as specified by CDF p(t).

        Args:
            lmd: beta distribution total parameter, lmd>0.
            phi: beta distribution mean parameter, 0<phi<1.
            cdf_start_end (tuple): CDF values of p(t) defining start and end
            times of disk build up

        Returns:
            type: Description of returned object.

        """
        a, b = self.get_beta_a_b_from_lmd_phi(lmd, phi)
        assert (a>1.)+(b>1) >= 1, "SFH is bimodal: increase lmd?'"
        age_bin_edges = self.cube.ssps.par_edges[1]
        age_loc = age_bin_edges[0]
        age_scale = age_bin_edges[-1] - age_bin_edges[0]
        beta = stats.beta(a, b,
                          loc=age_loc,
                          scale=age_scale)
        beta_cdf = beta.cdf(age_bin_edges)
        t_weights = beta_cdf[1:] - beta_cdf[:-1]
        dt = age_bin_edges[1:] - age_bin_edges[:-1]
        p_t = t_weights/dt
        t_start_end = beta.ppf(cdf_start_end)
        t_start, t_end = t_start_end
        idx_start_end = np.digitize(t_start_end, age_bin_edges)
        delta_t = t_end - t_start
        self.t_pars = dict(lmd=lmd,
                           phi=phi,
                           t_start=t_start,
                           t_end=t_end,
                           delta_t=delta_t,
                           idx_start_end=idx_start_end,
                           dt=dt)
        self.p_t = p_t

    def plot_sfh(self):
        """Plot the star-foramtion history p(t)
        """
        ax = plt.gca()
        ax.plot(self.cube.ssps.par_cents[1], self.p_t, '-o')
        ax.set_xlabel('Time [Gyr]')
        ax.set_ylabel('pdf')
        plt.tight_layout()
        plt.show()
        return

    def get_z_interpolation_grid(self,
                                 t_dep_lim=(0.1, 10.),
                                 n_t_dep=1000,
                                 n_z=1000):
        """Store a grid used for interpolating age-metallicity relations

        Args:
            t_dep_lim: (lo,hi) allowed values of depletion timescale in Gyr
            n_t_dep (int): number of steps to use for interpolating t_dep
            n_z (int): number of steps to use for interpolating metallicity

        """
        a = -0.689
        b = 1.899
        self.ahat = 10.**a
        self.bhat = b-1.
        z_max = self.ahat**(-1./self.bhat)
        # reduce z_max slightly to avoid divide by 0 warnings
        z_max *= 0.9999
        t_H = self.cube.ssps.par_edges[1][-1]
        log_t_dep_lim = np.log10(t_dep_lim)
        t_dep = np.logspace(*log_t_dep_lim, n_t_dep)
        z_lim = (z_max, 0., n_z)
        z = np.linspace(*z_lim, n_z)
        tt_dep, zz = np.meshgrid(t_dep, z, indexing='ij')
        t = t_H - tt_dep * zz/(1. - self.ahat * zz**self.bhat)
        self.t = t
        t_ssps = self.cube.ssps.par_cents[1]
        n_t_ssps = len(t_ssps)
        z_ssps = np.zeros((n_t_dep, n_t_ssps))
        for i in range(n_t_dep):
            z_ssps[i] = np.interp(t_ssps, t[i], z)
        self.z_t_interp_grid = dict(t=t_ssps,
                                    z=z_ssps,
                                    t_dep=t_dep)

    def evaluate_ybar(self):
        """Evaluate the noise-free data-cube

        Evaluate the integral
        ybar(x, omega) = int_{-inf}^{inf} s(omega-v ; t,z) P(t,v,x,z) dv dt dz
        where
        omega = ln(wavelength)
        s(omega ; t,z) are stored SSP templates
        This integral is a convolution over velocity v, which we evaluate using
        Fourier transforms (FT). FTs of SSP templates are stored in`ssps.FXw`
        while FTs of the velocity factor P(v|t,x) of the density P(t,v,x,z)
        are evaluated using the analytic expression of the FT of the normal
        distribution. Sets the result to `self.ybar`.

        """
        cube = self.cube
        ssps = cube.ssps
        # get P(t,x,z)
        P_txz = self.get_p_txz(density=False)
        # get FT of P(v|t,x)
        nl = ssps.FXw.shape[0]
        omega = np.linspace(0, np.pi, nl)
        omega /= ssps.dv
        omega = omega[:, np.newaxis, np.newaxis, np.newaxis]
        exponent = -1j*self.mu_v*omega - 0.5*(self.sig_v*omega)**2
        F_p_v_tx = np.exp(exponent)
        # get FT of SSP templates s(w;t,z)
        F_s_w_tz = ssps.FXw
        F_s_w_tz = np.reshape(F_s_w_tz, (-1,)+ssps.par_dims)
        # get FT of ybar
        args = P_txz, F_p_v_tx, F_s_w_tz
        F_ybar = np.einsum('txyz,wtxy,wzt->wxy', *args, optimize=True)
        ybar = np.fft.irfft(F_ybar, self.cube.ssps.n_fft, axis=0)
        self.ybar = ybar

    def linear_interpolate_t(self,
                             f_end,
                             f_start):
        """Linearly interpolate f against time, given boundary values

        Args:
            f_end: value of f at end of disk build up i.e. more recently
            f_start: value of f at start of disk build up i.e. more in the past

        Returns:
            array: f interpolated in age-bins of SSP models, set to constant
            values outside start/end times

        """
        t = self.cube.ssps.par_cents[1]
        delta_f = f_start - f_end
        t_from_start = t - self.t_pars['t_start']
        f = f_end + delta_f/self.t_pars['delta_t'] * t_from_start
        f[t < self.t_pars['t_start']] = f_end
        f[t > self.t_pars['t_end']] = f_start
        return f

    @abstractmethod
    def get_p_x_t(self, density=True, light_weighted=False):
        """Get p(x|t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        pass

    @abstractmethod
    def get_p_z_tx(self, density=True, light_weighted=False):
        """Get p(z|t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        pass

    @abstractmethod
    def get_p_v_tx(self, v_edg, density=True, light_weighted=False):
        """Get p(v|t,x)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        pass

    def get_p_t(self, density=True, light_weighted=False):
        """Get p(t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_t = self.p_t.copy()
            if density is False:
                p_t *= self.cube.ssps.delta_t
        else:
            p_tz = self.get_p_tz(density=False, light_weighted=True)
            p_t = np.sum(p_tz, 1)
            if density is True:
                ssps = self.cube.ssps
                p_t = p_t/ssps.delta_t
        return p_t

    def get_p_tv(self, v_edg, density=True, light_weighted=False):
        """Get p(t,v)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        if density is True:
            p_tvx *= self.cube.dx*self.cube.dy
        p_tv = np.sum(p_tvx, (-1,-2))
        return p_tv

    def get_p_tx(self, density=True, light_weighted=False):
        """Get p(t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_x_t = self.get_p_x_t(density=density, light_weighted=False)
            p_t = self.get_p_t(density=density, light_weighted=False)
            p_xt = p_x_t*p_t
            p_tx = np.einsum('xyt->txy', p_xt)
        else:
            p_txz = self.get_p_txz(density=density, light_weighted=True)
            if density is True:
                p_txz *= self.cube.ssps.delta_z
            p_tx = np.sum(p_txz, -1)
        return p_tx

    def get_p_tz(self, density=True, light_weighted=False):
        """Get p(t,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_txz = self.get_p_txz(density=density, light_weighted=light_weighted)
        if density is True:
            p_txz *= self.cube.dx*self.cube.dy
        p_tz = np.sum(p_txz, (1,2))
        return p_tz

    def get_p_tvx(self, v_edg, density=True, light_weighted=False):
        """Get p(t,v,x)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_tx = self.get_p_tx(density=density, light_weighted=False)
            p_v_tx = self.get_p_v_tx(v_edg,
                                     density=density,
                                     light_weighted=False)
            na = np.newaxis
            p_tvx = p_tx[:,na,:,:] * np.moveaxis(p_v_tx, 0, 1)
        else:
            p_tvxz = self.get_p_tvxz(v_edg,
                                     density=density,
                                     light_weighted=True)
            if density is True:
                p_tvxz *= self.cube.ssps.delta_z
            p_tvx = np.sum(p_tvxz, -1)
        return p_tvx

    def get_p_tvz(self, v_edg, density=True, light_weighted=False):
        """Get p(t,v,z)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvxz = self.get_p_tvxz(v_edg,
                                 density=density,
                                 light_weighted=light_weighted)
        if density is True:
            p_tvxz *= self.cube.dx*self.cube.dy
        p_tvz = np.sum(p_tvxz, (2,3))
        return p_tvz

    def get_p_txz(self, density=True, light_weighted=False):
        """Get p(t,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        p_tx = self.get_p_tx(density=density, light_weighted=False)
        p_z_tx = self.get_p_z_tx(density=density, light_weighted=False)
        p_txz = p_tx[:,:,:,na] * np.moveaxis(p_z_tx, 0, -1)
        if light_weighted:
            P_txz_mass_wtd = self.get_p_txz(density=False, light_weighted=False)
            light_weights = self.cube.ssps.light_weights[:,na,na,:]
            P_txz_light_wtd = P_txz_mass_wtd * light_weights
            normalisation = np.sum(P_txz_light_wtd)
            p_txz = p_txz * light_weights / normalisation
        return p_txz

    def get_p_tvxz(self, v_edg, density=True, light_weighted=False):
        """Get p(t,v,x,z)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        p_txz = self.get_p_txz(density=density, light_weighted=False)
        p_v_tx = self.get_p_v_tx(v_edg, density=density, light_weighted=False)
        p_v_txz = p_v_tx[:,:,:,:,na]
        p_tvxz = np.moveaxis(p_v_txz, 0, 1) * p_txz[:,na,:,:,:]
        if light_weighted:
            P_tvxz_mass_wtd = self.get_p_tvxz(v_edg,
                                              density=False,
                                              light_weighted=False)
            light_weights = self.cube.ssps.light_weights[:,na,na,na,:]
            P_tvxz_light_wtd = P_tvxz_mass_wtd * light_weights
            normalisation = np.sum(P_tvxz_light_wtd)
            p_tvxz = p_tvxz * light_weights / normalisation
        return p_tvxz

    def get_p_v(self, v_edg, density=True, light_weighted=False):
        """Get p(v)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        if density is True:
            p_tvx = (p_tvx.T * self.cube.ssps.delta_t).T
            p_tvx *= self.cube.dx*self.cube.dy
        p_v = np.sum(p_tvx, (0,2,3))
        return p_v

    def get_p_vx(self, v_edg, density=True, light_weighted=False):
        """Get p(v,x)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        if density is True:
            p_tvx = (p_tvx.T * self.cube.ssps.delta_t).T
        p_vx = np.sum(p_tvx, 0)
        return p_vx

    def get_p_vz(self, v_edg, density=True, light_weighted=False):
        """Get p(v,z)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_vxz = self.get_p_vxz(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        if density:
            p_vxz *= self.cube.dx*self.cube.dy
        p_vz = np.sum(p_vxz, (1,2))
        return p_vz

    def get_p_vxz(self, v_edg, density=True, light_weighted=False):
        """Get p(v,x,z)

        Args:
            v_edg : array of velocity-bin edges to evaluate the quantity
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvxz = self.get_p_tvxz(v_edg,
                                 density=density,
                                 light_weighted=light_weighted)
        if density:
            p_tvxz = (p_tvxz.T * self.cube.ssps.delta_t).T
        p_vxz = np.sum(p_tvxz, 0)
        return p_vxz

    def get_p_x(self, density=True, light_weighted=False):
        """Get p(x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_x_t = self.get_p_x_t(density=density)
            P_t = self.get_p_t(density=False)
            p_x = np.sum(p_x_t * P_t, -1)
        else:
            na = np.newaxis
            ssps = self.cube.ssps
            p_txz = self.get_p_txz(density=density, light_weighted=True)
            if density is False:
                p_x = np.sum(p_txz, (0,3))
            else:
                delta_tz = ssps.delta_t[:,na,na,na]*ssps.delta_z[na,na,na,:]
                p_x = np.sum(p_txz*delta_tz, (0,3))
        return p_x

    def get_p_xz(self, density=True, light_weighted=False):
        """Get p(x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_txz = self.get_p_txz(density=density, light_weighted=light_weighted)
        if density:
            p_txz = (p_txz.T * self.cube.ssps.delta_t).T
        p_xz = np.sum(p_txz, 0)
        return p_xz

    def get_p_z(self, density=True, light_weighted=False):
        """Get p(z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        if light_weighted is False:
            p_z_tx = self.get_p_z_tx(density=density)
            P_x_t = self.get_p_x_t(density=False) # to marginalise, must be a probabilty
            P_x_t = np.einsum('xyt->txy', P_x_t)
            P_x_t = P_x_t[na,:,:,:]
            P_t = self.get_p_t(density=False) # to marginalise, must be a probabilty
            P_t = P_t[na,:,na,na]
            p_z = np.sum(p_z_tx * P_x_t * P_t, (1,2,3))
        else:
            p_tz = self.get_p_tz(density=density, light_weighted=True)
            if density is False:
                p_z = np.sum(p_tz, 0)
            else:
                ssps = self.cube.ssps
                delta_t = ssps.delta_t[:,na]
                p_z = np.sum(p_tz*delta_t, 0)
        return p_z

    def get_p(self,
              which_dist,
              density=True,
              light_weighted=False,
              v_edg=None):
        """Evaluate population-kinematic densities of this galaxy component

        Evaluate marginal or conditional densities over: stellar age (t), 2D
        position (x), velocity (v) and metallicity (z). Argument `which_dist`
        specifies which distribution to evaluate where underscore represents
        conditioning e.g.
        - `which_dist = 'tv'` --> p(t,v),
        - `which_dist = 'tz_x'` --> p(t,z|x) etc...
        Variables in `which_dist` must be provided in alphabetical order (on
        either side of the underscore if present).

        Args:
            which_dist (string): which density to evaluate
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity
            v_edg (array): array of velocity-bin edged, required only if 'v' in
                `which_dist`.

        Returns:
            array: the desired distribution. Array dimensions correspond to the
                order of variables as provided in `which_dist` string e.g.
                `which_dist = tz_x` returns p(t,z|x) as a 4D array with
                dimensions corresponding to [t,z,x1,x2].

        """
        # TODO: error catching if the order of variables is not alphabetical
        is_conditional = '_' in which_dist
        if is_conditional:
            p = self.get_conditional_distribution(
                which_dist,
                density=density,
                light_weighted=light_weighted,
                v_edg=v_edg)
        else:
            p = self.get_marginal_distribution(
                which_dist,
                density=density,
                light_weighted=light_weighted,
                v_edg=v_edg)
        return p

    def get_marginal_distribution(self,
                                  which_dist,
                                  density=True,
                                  v_edg=None,
                                  light_weighted=False):
        """Evaluate component-wise marginal distributions

        Args:
            which_dist (string): which density to evaluate (this must be
                marginal, not conditional).
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            v_edg (array): array of velocity-bin edges.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array: dimensions align with variables as listed in `which_dist`

        """
        # TODO: check that `which_dist` has no underscore and is alphabetical
        p_func = getattr(self, 'get_p_'+which_dist)
        if 'v' in which_dist:
            p = p_func(v_edg=v_edg,
                       density=density,
                       light_weighted=light_weighted)
        else:
            p = p_func(density=density, light_weighted=light_weighted)
        return p

    def get_conditional_distribution(self,
                                     which_dist,
                                     light_weighted=False,
                                     density=True,
                                     v_edg=None):
        """Get conditional distributions

        This is intended to be called only by the `get_p` wrapper method - see
        that docstring for more info.

        Args:
        which_dist (string): which density to evaluate
        density (bool): whether to return probabilty density (True) or the
        volume-element weighted probabilty (False)
        v_edg (array): array of velocity-bin edges

        Returns:
        array: the desired distribution.

        """
        assert '_' in which_dist
        dist, marginal = which_dist.split('_')
        # if the conditional distribution is hard coded, then use that...
        if hasattr(self, 'get_p_'+which_dist):
            p_func = getattr(self, 'get_p_'+which_dist)
            if 'v' in dist:
                 p_conditional = p_func(v_edg,
                                        density=density,
                                        light_weighted=light_weighted)
            else:
                 p_conditional = p_func(density=density,
                                        light_weighted=light_weighted)
        # ... otherwise compute conditional = joint/marginal
        else:
            joint = ''.join(sorted(dist+marginal))
            kwargs = {'density':False,
                      'v_edg':v_edg,
                      'light_weighted':light_weighted}
            p_joint = self.get_p(joint, **kwargs)
            p_marginal = self.get_p(marginal, **kwargs)
            # if x is in joint/marginalal, repalace it with xy to account for the
            # fact that x stands for 2D positon (x,y)
            joint = joint.replace('x', 'xy')
            marginal = marginal.replace('x', 'xy')
            # for each entry in the marginal, find its position in the joint
            old_pos_in_joint = [joint.find(m0) for m0 in marginal]
            # move the marginal variables to the far right of the joint
            n_marginal = len(marginal)
            new_pos_in_joint = [-(i+1) for i in range(n_marginal)][::-1]
            p_joint = np.moveaxis(p_joint, old_pos_in_joint, new_pos_in_joint)
            # get the conditional probability
            p_conditional = p_joint/p_marginal
            if density:
                dvol = self.construct_volume_element(which_dist, v_edg=v_edg)
                p_conditional = p_conditional/dvol
        return p_conditional

    def construct_volume_element(self,
                                 which_dist,
                                 v_edg=None):
        """Construct volume element for converting densities to probabilties

        Args:
            which_dist (string): which density to evaluate
            v_edg (array): array of velocity-bin edges

        Returns:
            array: The volume element with correct shape for `which_dist`

        """
        dist_is_conditional = '_' in which_dist
        if dist_is_conditional:
            dist_string, marginal_string = which_dist.split('_')
            marginal_string = marginal_string.replace('x', 'xy')
        else:
            dist_string = which_dist
        dist_string = dist_string.replace('x', 'xy')
        count = 0
        ndim = len(dist_string)
        dvol = np.ones([1 for i in range(ndim)])
        na = np.newaxis
        slc = slice(0,None)
        for var in dist_string:
            if var=='t':
                da = self.cube.ssps.delta_t
            elif var=='v':
                da = v_edg[1:] - v_edg[:-1]
            elif var=='x':
                da = np.array([self.dx])
            elif var=='y':
                da = np.array([self.dy])
            elif var=='z':
                da = self.cube.ssps.delta_z
            idx = tuple([na for i in range(count)])
            idx = idx + (slc,)
            idx = idx + tuple([na for i in range(ndim-count-1)])
            dvol = dvol * da[idx]
            count += 1
        idx = tuple([slc for i in range(dvol.ndim)])
        if dist_is_conditional:
            idx = idx + tuple([na for i in range(len(marginal_string))])
        dvol = dvol[idx]
        return dvol


# end
