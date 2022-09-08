import numpy as np
from scipy import stats, special
from abc import ABC, abstractmethod
from . import base

class parametricComponent(base.baseComponent):
    """Abstract base class to reprepsent parametrised galaxy components

    Density factors as p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x), where:
    - p(t) is a beta distribution,
    - p(x|t) is some parameterised function where the component has a specified
    `center` and `rotation` relative to the cube. Sub-classes should implement
    specific cases in their `set_p_x_t` method
    - p(v|t,x) = Normal(v ; mu_v(t,x), sig_v(t,x)). Sub-classes should implement
    specific cases in their methods `set_mu_v` and `set_sig_v`
    - p(z|t,x) = Normal(z ; mu_z(t, t_dep(x)), sig_z(t, t_dep(x))) i.e.
    metallicity depends on a spatially varying depletion timescale t_dep(x). The
    spcific form of mu_z(t,t_dep) and mu_z(t,t_dep) are taken from a chemical
    evolution model from eqns 3-10 of Zhu, van de Venn, Leaman et al 2020

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.
        v_edg (array): velocity bin edges used to evaluate densities.

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
        self.v_edg = cube.v_edg

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
        if light_weighted is False:
            p_x_t = self.p_x_t.copy()
            if density is False:
                p_x_t *= (self.cube.dx * self.cube.dy)
        else:
            p_txz = self.get_p_txz(density=density, light_weighted=True)
            if density is True:
                ssps = self.cube.ssps
                p_txz = p_txz * ssps.delta_z
            p_tx = np.sum(p_txz, -1)
            p_t = self.get_p_t(density=density, light_weighted=True)
            p_x_t = (p_tx.T/p_t).T
            p_x_t = np.einsum('txy->xyt', p_x_t)
        return p_x_t

    @abstractmethod
    def set_mu_v(self):
        """Set spatially and age dependent mean velocity

        Should set an attribute `self.mu_v`, a 3D array of size [nt, nx1, nx2]
        """
        pass

    @abstractmethod
    def set_sig_v(self):
        """Set spatially and age dependent velocity dispersions

        Should set an attribute `self.sig_v`, a 3D array of size [nt, nx1, nx2]
        """
        pass

    def get_p_v_tx(self, density=True, light_weighted=False):
        """Get p(v|t,x)

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
            v_edg = self.v_edg[:, na, na, na]
            norm = stats.norm(loc=self.mu_v, scale=self.sig_v)
            p_v_tx = norm.cdf(v_edg[1:]) - norm.cdf(v_edg[:-1])
            if density is True:
                dv = v_edg[1:] - v_edg[:-1]
                p_v_tx /= dv
        else:
            p_tvxz = self.get_p_tvxz(density=True, light_weighted=True)
            if density is False:
                dv = self.v_edg[1:] - self.v_edg[:-1]
                dv = dv[na, :, na, na, na]
                p_tvxz = p_tvxz*dv
            ssps = self.cube.ssps
            p_tvx = np.sum(p_tvxz*ssps.delta_z, -1)
            p_x_t = self.get_p_x_t(density=True, light_weighted=True)
            p_t = self.get_p_t(density=True, light_weighted=True)
            p_xt = p_x_t * p_t
            p_tx = np.einsum('xyt->txy', p_xt)
            p_tx = p_tx[:, na, :, :]
            p_v_tx = p_tvx/p_tx
            p_v_tx = np.einsum('tvxy->vtxy', p_v_tx)
        return p_v_tx

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

    @abstractmethod
    def set_t_dep(self):
        """Set spatially varying depletion timescale

        Should set an attribute `self.t_dep` - a 2D array of depletion time at
        2D position

        """
        pass

    def set_p_z_tx(self):
        """Set p(z|t,x) given enrichment model and spatially varying t_dep

        Evaluates the chemical evolution model defined in equations 3-10 of
        Zhu et al 2020, parameterised by a spatially varying depletion
        timescale stored by `set_t_dep`.

        """
        del_t_dep = self.t_dep[:,:,np.newaxis] - self.z_t_interp_grid['t_dep']
        abs_del_t_dep = np.abs(del_t_dep)
        idx_t_dep = np.argmin(abs_del_t_dep, axis=-1)
        self.idx_t_dep = idx_t_dep
        idx_t_dep = np.ravel(idx_t_dep)
        z_mu = self.z_t_interp_grid['z'][idx_t_dep, :]
        z_mu = np.reshape(z_mu, (self.cube.nx, self.cube.ny, -1))
        z_mu = np.moveaxis(z_mu, -1, 0)
        z_sig2 = self.ahat * z_mu**self.bhat
        log_z_edg = self.cube.ssps.par_edges[0]
        del_log_z = log_z_edg[1:] - log_z_edg[:-1]
        x_xsun = 1. # i.e. assuming galaxy has hydrogen mass fraction = solar
        lin_z_edg = 10.**log_z_edg * x_xsun
        nrm = stats.norm(loc=z_mu, scale=z_sig2**0.5)
        lin_z_edg = lin_z_edg[:, np.newaxis, np.newaxis, np.newaxis]
        cdf_z_tx = nrm.cdf(lin_z_edg)
        p_z_tx = cdf_z_tx[1:] - cdf_z_tx[:-1]
        nrm = np.sum(p_z_tx.T * del_log_z, -1).T
        p_z_tx /= nrm
        self.p_z_tx = p_z_tx

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
        if light_weighted is False:
            p_z_tx = self.p_z_tx.copy()
        else:
            p_txz = self.get_p_txz(density=True, light_weighted=True)
            p_tx = self.get_p_tx(density=True, light_weighted=True)
            p_z_tx = (p_txz.T/p_tx.T).T
            p_z_tx = np.einsum('txyz->ztxy', p_z_tx)
        if density is False:
            dz = self.cube.ssps.delta_z
            na = np.newaxis
            dz = dz[:, na, na, na]
            p_z_tx *= dz
        return p_z_tx

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

    def get_p_tv(self, density=True, light_weighted=False):
        """Get p(t,v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(density=density, light_weighted=light_weighted)
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

    def get_p_tvx(self, density=True, light_weighted=False):
        """Get p(t,v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_tx = self.get_p_tx(density=density, light_weighted=False)
            p_v_tx = self.get_p_v_tx(density=density, light_weighted=False)
            na = np.newaxis
            p_tvx = p_tx[:,na,:,:] * np.moveaxis(p_v_tx, 0, 1)
        else:
            p_tvxz = self.get_p_tvxz(density=density, light_weighted=True)
            if density is True:
                p_tvxz *= self.cube.ssps.delta_z
            p_tvx = np.sum(p_tvxz, -1)
        return p_tvx

    def get_p_tvz(self, density=True, light_weighted=False):
        """Get p(t,v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvxz = self.get_p_tvxz(density=density, light_weighted=light_weighted)
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

    def get_p_tvxz(self, density=True, light_weighted=False):
        """Get p(t,v,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        p_txz = self.get_p_txz(density=density, light_weighted=False)
        p_v_tx = self.get_p_v_tx(density=density, light_weighted=False)
        p_v_txz = p_v_tx[:,:,:,:,na]
        p_tvxz = np.moveaxis(p_v_txz, 0, 1) * p_txz[:,na,:,:,:]
        if light_weighted:
            P_tvxz_mass_wtd = self.get_p_tvxz(density=False,
                                              light_weighted=False)
            light_weights = self.cube.ssps.light_weights[:,na,na,na,:]
            P_tvxz_light_wtd = P_tvxz_mass_wtd * light_weights
            normalisation = np.sum(P_tvxz_light_wtd)
            p_tvxz = p_tvxz * light_weights / normalisation
        return p_tvxz

    def get_p_v(self, density=True, light_weighted=False):
        """Get p(v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(density=density, light_weighted=light_weighted)
        if density is True:
            p_tvx = (p_tvx.T * self.cube.ssps.delta_t).T
            p_tvx *= self.cube.dx*self.cube.dy
        p_v = np.sum(p_tvx, (0,2,3))
        return p_v

    def get_p_vx(self, density=True, light_weighted=False):
        """Get p(v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvx = self.get_p_tvx(density=density, light_weighted=light_weighted)
        if density is True:
            p_tvx = (p_tvx.T * self.cube.ssps.delta_t).T
        p_vx = np.sum(p_tvx, 0)
        return p_vx

    def get_p_vz(self, density=True, light_weighted=False):
        """Get p(v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_vxz = self.get_p_vxz(density=density, light_weighted=light_weighted)
        if density:
            p_vxz *= self.cube.dx*self.cube.dy
        p_vz = np.sum(p_vxz, (1,2))
        return p_vz

    def get_p_vxz(self, density=True, light_weighted=False):
        """Get p(v,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tvxz = self.get_p_tvxz(density=density, light_weighted=light_weighted)
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

    def get_E_v_x(self, light_weighted=False):
        """Get mean velocity map E[p(v|x)]

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        P_t = self.get_p_t(density=False, light_weighted=light_weighted)
        E_v_x = np.sum((P_t*self.mu_v.T).T, 0)
        return E_v_x

    def get_jth_central_moment_v_x(self, j, light_weighted=False):
        """Get j'th central moment of velocity map E[p((v-mu_v)^j|x)]

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        P_t = self.get_p_t(density=False, light_weighted=light_weighted)
        mu = self.get_E_v_x()
        k = np.arange(0, j+1, 2)
        tmp1 = special.comb(j, k)
        na = np.newaxis
        tmp2 = (self.mu_v - mu)[na,:,:,:]**(j-k[:,na,na,na])
        tmp3 = 1.*P_t
        tmp4 = self.sig_v[na,:,:,:]**k[:,na,na,na]
        tmp5 = special.factorial2(k-1)
        muj_v_x = np.einsum('k,ktxy,t,ktxy,k->xy',
                            special.comb(j, k),
                            (self.mu_v - mu)[na,:,:,:]**(j-k[:,na,na,na]),
                            P_t,
                            self.sig_v[na,:,:,:]**k[:,na,na,na],
                            special.factorial2(k-1))
        return muj_v_x

    def plot_density(self,
                     vmin=0.1,
                     vmax=3.,
                     show_every_nth_time=4):
        """Plot maps of the spatial density p(x|t) at several timesteps

        Plot the density between the start/end times of disk growth, which
        depends on the CDF of p(t). Skip every N steps between these points.

        Args:
            vmin: minimum velocity for colormap
            vmax: maximum velocity for colormap
            show_every_nth_time (int): number of timesteps to skip between plots

        """
        t_idx_list = np.arange(*self.t_pars['idx_start_end'],
                               show_every_nth_time)
        t_idx_list = t_idx_list[::-1]
        kw_imshow = {'cmap':plt.cm.gist_heat,
                     'norm':LogNorm(vmin=vmin, vmax=vmax)}
        for t_idx in t_idx_list:
            t = self.cube.ssps.par_cents[1][t_idx]
            img = self.cube.imshow(self.p_x_t[:,:,t_idx], **kw_imshow)
            plt.gca().set_title(f't={t}')
            plt.tight_layout()
            plt.show()
        return

    def plot_t_dep(self):
        """Plot map of depletion timescale used for chemical enrichment
        """
        kw_imshow = {'cmap':plt.cm.jet}
        img = self.cube.imshow(self.t_dep,
                              colorbar_label='$t_\\mathrm{dep}$',
                              **kw_imshow)
        plt.tight_layout()
        plt.show()
        return

    def plot_mu_v(self,
                  show_every_nth_time=4,
                  vmax=None):
        """Plot maps of the mean velocity E[p(v|t,x)] at several timesteps

        Plot the map between the start/end times of disk growth, which
        depends on the CDF of p(t). Skip every N steps between these points.

        Args:
            vmax: maximum velocity for colormap
            show_every_nth_time (int): number of timesteps to skip between plots

        """
        if vmax is None:
            vmax = np.max(np.abs(self.mu_v_pars['vmax_lims']))
        cube = self.cube
        t_idx_list = np.arange(*self.t_pars['idx_start_end'],
                               show_every_nth_time)
        t_idx_list = t_idx_list[::-1]
        kw_imshow = {'vmin':-vmax, 'vmax':vmax}
        for t_idx in t_idx_list:
            t = self.cube.ssps.par_cents[1][t_idx]
            self.cube.imshow(self.mu_v[t_idx,:,:], **kw_imshow)
            plt.gca().set_title(f't={t}')
            plt.tight_layout()
            plt.show()

    def plot_sig_v(self,
                   show_every_nth_time=4,
                   vmin=None,
                   vmax=None):
        """Plot maps of the dispersion of p(v|t,x) at several timesteps

        Plot the map between the start/end times of disk growth, which
        depends on the CDF of p(t). Skip every N steps between these points.

        Args:
            vmax: minimum velocity for colormap
            vmax: maximum velocity for colormap
            show_every_nth_time (int): number of timesteps to skip between plots

        """
        cube = self.cube
        t_idx_list = np.arange(*self.t_pars['idx_start_end'],
                               show_every_nth_time)
        t_idx_list = t_idx_list[::-1]
        sigs = np.concatenate((
            self.sig_v_pars['sig_v_in_lims'],
            self.sig_v_pars['sig_v_out_lims']
            ))
        if vmin is None:
            vmin = np.min(sigs)
        if vmax is None:
            vmax = np.max(sigs)
        kw_imshow = {'cmap':plt.cm.jet,
                     'vmin':vmin,
                     'vmax':vmax}
        for t_idx in t_idx_list:
            t = cube.ssps.par_cents[1][t_idx]
            cube.imshow(self.sig_v[t_idx,:,:], **kw_imshow)
            plt.gca().set_title(f't={t}')
            plt.tight_layout()
            plt.show()

# end
