import numpy as np
from scipy import stats, special
from abc import abstractmethod
from . import base

class ParametricComponent(base.Component):
    """Abstract class for parametrised galaxy components

    The mass-weighted density factorises as

    p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x)

    where:

    - p(t) : beta distribution (see `set_p_t`),
    - p(x|t) : function to be implemented in subclass's `set_p_x_t` method    
    - p(v|t,x) = Normal(v ; mu_v(t,x), sig_v(t,x)) where subclasses provide
      specific implementations in their `set_mu_v` and `set_sig_v` methods,
    - p(z|t,x) = Normal(z ; mu_z(t, t_dep(x)), sig_z(t, t_dep(x))) i.e. 
      chemical enrichment depends on a spatially varying depletion timescale 
      t_dep(x) to be implemented in the `set_t_dep` method of subclass.
    
    The functions  mu_z(t,t_dep) and sig_z(t,t_dep) are from equations 3-10 of 
    Zhu et al. 20.

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

    def linear_interpolate_t(self,
                             f_end,
                             f_start):
        """Linearly interpolate f against t given boundary [f_start, f_end]

        Args:
            f_end: value of f at end of disk build up i.e. more recently
            f_start: value of f at start of disk build up i.e. more in the past

        Returns:
            array: f interpolated to the t array of SSP models, set to constant
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
        """Set the mass-weighted star formation history p(t)

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
        if (a>1.)+(b>1) < 1:
            raise ValueError("SFH is bimodal - increase lmd?'")
        age_bin_edges = self.cube.ssps.par_edges[1]
        age_loc = age_bin_edges[0]
        age_scale = age_bin_edges[-1] - age_bin_edges[0]
        beta = stats.beta(a, b,
                          loc=age_loc,
                          scale=age_scale)
        t_start_end = beta.ppf(cdf_start_end)
        t_start, t_end = t_start_end
        idx_start_end = np.digitize(t_start_end, age_bin_edges)
        delta_t = t_end - t_start
        self.t_pars = dict(lmd=lmd,
                           phi=phi,
                           t_start=t_start,
                           t_end=t_end,
                           delta_t=delta_t,
                           idx_start_end=idx_start_end)
        t_edges = self.cube.ssps.par_edges[1]
        log_cdf_t = beta.logcdf(t_edges)
        tmp = np.array([log_cdf_t[1:], log_cdf_t[:-1]])
        tmp = special.logsumexp(tmp.T, -1, [1,-1]).T
        log_dt = np.log(self.cube.construct_volume_element('t'))
        self.log_p_t = tmp - log_dt
        self.p_t = np.exp(self.log_p_t)

    @abstractmethod
    def set_p_x_t(self):
        """Set the age dependent spatial density p(x|t)

        Should set an attribute `self.log_p_x_t` 3D array of the log density of
        size [nx1,nx2,nt]

        """
        pass

    def get_log_p_x_t(self, density=True, light_weighted=False):
        """Get log p(x|t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_x_t = self.log_p_x_t.copy()
        else:
            log_p_txz = self.get_log_p_txz(density=True, light_weighted=True)
            log_dz = np.log(self.cube.construct_volume_element('z'))
            log_p_tx = special.logsumexp(log_p_txz + log_dz, -1)
            log_p_t = self.get_log_p_t(density=True, light_weighted=True)
            log_p_x_t = (log_p_tx.T - log_p_t).T
            log_p_x_t = np.moveaxis(log_p_x_t, 0, -1)
        if density is False:
            log_p_x_t += np.log(self.cube.dx) + np.log(self.cube.dy)
        return log_p_x_t

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

    def get_log_p_v_tx(self, density=True, light_weighted=False):
        """Get log p(v|t,x)

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
            v_edg = self.cube.v_edg[:, na, na, na]
            norm = stats.norm(loc=self.mu_v, scale=self.sig_v)
            log_cdf_0 = norm.logcdf(v_edg[:-1])
            log_cdf_1 = norm.logcdf(v_edg[1:])
            log_cdf = np.array([log_cdf_1, log_cdf_0]).T
            log_p_v_tx = special.logsumexp(log_cdf, -1, b=[1,-1]).T
        else:
            log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=True)
            log_p_tvx = special.logsumexp(log_p_tvxz, -1)
            log_p_tx = special.logsumexp(log_p_tvx, 1)
            na = np.newaxis
            log_p_v_tx = log_p_tvx - log_p_tx[:,na,:,:]
            log_p_v_tx = np.moveaxis(log_p_v_tx, 0, 1)
        if density is True:
            dv = self.cube.construct_volume_element('v')
            log_p_v_tx = (log_p_v_tx.T - np.log(dv)).T
        return log_p_v_tx

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

        Should set an attribute `self.t_dep`, a 2D array of depletion time vs 
        2D position, of size [nx1, nx2] and end with `self.set_p_z_tx()`.

        """
        pass

    def evaluate_chemical_enrichment_model_base(self, t_dep):
        """Evaluate chemical enrichment model for a t_dep(x) 

        Evaluates the chemical evolution model defined in equations 3-10 of
        Zhu et al 2020, parameterised by a spatially varying depletion
        timescale created by `set_t_dep`.

        """
        del_t_dep = t_dep[:,:,np.newaxis] - self.z_t_interp_grid['t_dep']
        abs_del_t_dep = np.abs(del_t_dep)
        idx_t_dep = np.argmin(abs_del_t_dep, axis=-1)
        z_mu = self.z_t_interp_grid['z'][np.ravel(idx_t_dep), :]
        z_mu = np.reshape(z_mu, (self.cube.nx, self.cube.ny, -1))
        z_mu = np.moveaxis(z_mu, -1, 0)
        z_sig2 = self.ahat * z_mu**self.bhat
        log_z_edg = self.cube.ssps.par_edges[0]
        x_xsun = 1. # i.e. assuming galaxy has hydrogen mass fraction = solar
        lin_z_edg = 10.**log_z_edg * x_xsun
        nrm = stats.norm(loc=z_mu, scale=z_sig2**0.5)
        lin_z_edg = lin_z_edg[:, np.newaxis, np.newaxis, np.newaxis]
        log_cdf_z_tx = nrm.logcdf(lin_z_edg)
        tmp = np.array([log_cdf_z_tx[1:], log_cdf_z_tx[:-1]])
        tmp = special.logsumexp(tmp.T, -1, b=[1,-1]).T
        log_dz = np.log(self.cube.construct_volume_element('z'))
        log_norm = special.logsumexp(tmp.T + log_dz, -1).T
        log_p_z_tx = tmp - log_norm
        p_z_tx = np.exp(log_p_z_tx)
        return log_p_z_tx, p_z_tx

    def set_p_z_tx(self):
        """Set p(z|t,x) given enrichment model and spatially varying t_dep

        Evaluates the chemical evolution model defined in equations 3-10 of
        Zhu et al 2020, parameterised by a spatially varying depletion
        timescale created by `set_t_dep`.

        """
        log_p_z_tx, p_z_tx = self.evaluate_chemical_enrichment_model_base(
            self.t_dep
        )
        self.log_p_z_tx = log_p_z_tx
        self.p_z_tx = p_z_tx

    def evaluate_chemical_enrichment_model(self, t_dep):
        """Evaluate chemical enrichment model for a single t_dep

        Hacky wrapper around `self.evaluate_chemical_enrichment_model`

        """
        # pass array full of single provided t_dep
        log_p_z_tx, p_z_tx = self.evaluate_chemical_enrichment_model_base(
            np.full((self.cube.nx, self.cube.ny), t_dep)
        )
        # take p_z_t from (x,y) = (0,0) since t_dep is constant  
        p_z_t = p_z_tx[:,:,0,0]
        return p_z_t

    def get_log_p_z_tx(self, density=True, light_weighted=False):
        """Get log p(z|t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_z_tx = self.log_p_z_tx.copy() # = density
        else:
            log_p_txz = self.get_log_p_txz(density=True, light_weighted=True)
            log_dz = np.log(self.cube.construct_volume_element('z'))
            log_p_tx = special.logsumexp(log_p_txz + log_dz, -1)
            log_p_z_tx = (log_p_txz.T - log_p_tx.T).T
            log_p_z_tx = np.moveaxis(log_p_z_tx, -1, 0)
        if density is False:
            log_dz = np.log(self.cube.construct_volume_element('z'))
            log_p_z_tx = (log_p_z_tx.T + log_dz).T
        return log_p_z_tx

    def evaluate_ybar(self):
        """Evaluate the datacube for this component

        Evaluate the integral

        ybar(x, omega) = int s(omega-v ; t,z) p(t,x,z) p(v|t,x) dv dt dz
        
        This integral is a convolution over velocity v, which we evaluate using
        Fourier transforms (FT). FTs of SSP templates are stored in`ssps.FXw`
        while FTs of the velocity factor p(v|t,x) are evaluated using
        analytic expressions of the FT of the normal distribution. Sets the 
        result to `self.ybar`.

        """
        cube = self.cube
        ssps = cube.ssps
        # get P(t,x,z)
        P_txz = self.get_p('txz', density=False)
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

    def get_log_p_t(self, density=True, light_weighted=False):
        """Get log p(t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_t = self.log_p_t.copy()
        else:
            log_p_tz = self.get_log_p_tz(density=True, light_weighted=True)
            log_dz = np.log(self.cube.construct_volume_element('z'))
            log_p_t = special.logsumexp(log_p_tz + log_dz, 1)
        if density is False:
            log_p_t += np.log(self.cube.construct_volume_element('t'))
        return log_p_t

    def get_log_p_tv(self, density=True, light_weighted=False):
        """Get log p(t,v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvx = self.get_log_p_tvx(
            density=density,
            light_weighted=light_weighted)
        if density is True:
            log_p_tvx += np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_tv = special.logsumexp(log_p_tvx, (-1,-2))
        return log_p_tv

    def get_log_p_tx(self, density=True, light_weighted=False):
        """Get log p(t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_x_t = self.get_log_p_x_t(
                density=density,
                light_weighted=False)
            log_p_t = self.get_log_p_t(density=density, light_weighted=False)
            log_p_tx = log_p_x_t + log_p_t
            log_p_tx = np.moveaxis(log_p_tx, -1, 0)
        else:
            log_p_txz = self.get_log_p_txz(density=density, light_weighted=True)
            if density is True:
                log_p_txz += np.log(self.cube.construct_volume_element('z'))
            log_p_tx = special.logsumexp(log_p_txz, -1)
        return log_p_tx

    def get_log_p_tz(self, density=True, light_weighted=False):
        """Get log p(t,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_txz = self.get_log_p_txz(density=density,
                                       light_weighted=light_weighted)
        if density is True:
            log_p_txz += np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_tz = special.logsumexp(log_p_txz, (1,2))
        return log_p_tz

    def get_log_p_tvx(self, density=True, light_weighted=False):
        """Get log p(t,v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_tx = self.get_log_p_tx(density=density,
                                         light_weighted=False)
            log_p_v_tx = self.get_log_p_v_tx(density=density,
                                             light_weighted=False)
            na = np.newaxis
            log_p_tvx = log_p_tx[:,na,:,:] + np.moveaxis(log_p_v_tx, 0, 1)
        else:
            log_p_tvxz = self.get_log_p_tvxz(density=density,
                                             light_weighted=True)
            if density is True:
                log_p_tvxz += np.log(self.cube.construct_volume_element('z'))
            log_p_tvx = special.logsumexp(log_p_tvxz, -1)
        return log_p_tvx

    def get_log_p_tvz(self, density=True, light_weighted=False):
        """Get log p(t,v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self.get_log_p_tvxz(density=density,
                                         light_weighted=light_weighted)
        if density is True:
            log_p_tvxz += np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_tvz = special.logsumexp(log_p_tvxz, (2,3))
        return log_p_tvz

    def get_log_p_txz(self, density=True, light_weighted=False):
        """Get log p(t,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        log_p_tx = self.get_log_p_tx(density=density, light_weighted=False)
        log_p_z_tx = self.get_log_p_z_tx(density=density, light_weighted=False)
        log_p_txz = log_p_tx[:,:,:,na] + np.moveaxis(log_p_z_tx, 0, -1)
        if light_weighted:
            log_P_txz_mass_wtd = self.get_log_p_txz(
                density=False,
                light_weighted=False)
            light_weights = self.cube.ssps.light_weights[:,na,na,:]
            log_lw = np.log(light_weights)
            log_P_txz_light_wtd = log_P_txz_mass_wtd + log_lw
            log_normalisation = special.logsumexp(log_P_txz_light_wtd)
            log_p_txz = log_p_txz + log_lw - log_normalisation
        return log_p_txz

    def get_log_p_tvxz(self, density=True, light_weighted=False):
        """Get log p(t,v,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        na = np.newaxis
        log_p_txz = self.get_log_p_txz(density=density, light_weighted=False)
        log_p_v_tx = self.get_log_p_v_tx(density=density, light_weighted=False)
        log_p_v_txz = log_p_v_tx[:,:,:,:,na]
        log_p_tvxz = np.moveaxis(log_p_v_txz, 0, 1) + log_p_txz[:,na,:,:,:]
        if light_weighted:
            log_P_tvxz_mass_wtd = self.get_log_p_tvxz(
                density=False,
                light_weighted=False)
            light_weights = self.cube.ssps.light_weights[:,na,na,na,:]
            log_lw = np.log(light_weights)
            log_P_tvxz_light_wtd = log_P_tvxz_mass_wtd + log_lw
            log_normalisation = special.logsumexp(log_P_tvxz_light_wtd)
            log_p_tvxz = log_p_tvxz + log_lw - log_normalisation
        return log_p_tvxz

    def get_log_p_v(self, density=True, light_weighted=False):
        """Get log p(v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvx = self.get_log_p_tvx(density=density,
                                       light_weighted=light_weighted)
        if density is True:
            log_dt = np.log(self.cube.construct_volume_element('t'))
            log_p_tvx = (log_p_tvx.T + log_dt).T
            log_p_tvx += np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_tvx = special.logsumexp(log_p_tvx, (0,2,3))
        return log_p_tvx

    def get_log_p_vx(self, density=True, light_weighted=False):
        """Get log p(v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvx = self.get_log_p_tvx(
            density=density,
            light_weighted=light_weighted)
        if density is True:
            log_dt = np.log(self.cube.construct_volume_element('t'))
            log_p_tvx = (log_p_tvx.T + log_dt).T
        log_p_tvx = special.logsumexp(log_p_tvx, 0)
        return log_p_tvx

    def get_log_p_vz(self, density=True, light_weighted=False):
        """Get log p(v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_vxz = self.get_log_p_vxz(density=density,
                                       light_weighted=light_weighted)
        if density:
            log_p_vxz += np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_vz = special.logsumexp(log_p_vxz, (1,2))
        return log_p_vz

    def get_log_p_vxz(self, density=True, light_weighted=False):
        """Get log p(v,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self.get_log_p_tvxz(
            density=density,
            light_weighted=light_weighted)
        if density:
            log_dt = np.log(self.cube.construct_volume_element('t'))
            log_p_tvxz = (log_p_tvxz.T + log_dt).T
        log_p_vxz = special.logsumexp(log_p_tvxz, 0)
        return log_p_vxz

    def get_log_p_x(self, density=True, light_weighted=False):
        """Get log p(x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            log_p_x_t = self.get_log_p_x_t(density=density)
            log_P_t = self.get_log_p_t(density=False)
            log_p_x = special.logsumexp(log_p_x_t + log_P_t, -1)
        else:
            log_p_txz = self.get_log_p_txz(density=density, light_weighted=True)
            if density is True:
                log_dtdz = np.log(self.cube.construct_volume_element('tz'))
                na = np.newaxis
                log_p_txz += log_dtdz[:,na,na,:]
            log_p_x = special.logsumexp(log_p_txz, (0,3))
        return log_p_x

    def get_log_p_xz(self, density=True, light_weighted=False):
        """Get log p(x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_txz = self.get_log_p_txz(density=density,
                                       light_weighted=light_weighted)
        if density:
            log_dt = np.log(self.cube.construct_volume_element('t'))
            log_p_txz = (log_p_txz.T + log_dt).T
        log_p_xz = special.logsumexp(log_p_txz, 0)
        return log_p_xz

    def get_log_p_z(self, density=True, light_weighted=False):
        """Get log p(z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tz = self.get_log_p_tz(density=density,
                                     light_weighted=light_weighted)
        if density:
            log_dt = np.log(self.cube.construct_volume_element('t'))
            log_p_tz = (log_p_tz.T + log_dt).T
        log_p_z = special.logsumexp(log_p_tz, 0)
        return log_p_z

    def get_central_moment_v_tx(self, j, light_weighted=False):
        """Get j'th central moment of p(v|t,x)

        Uses analytic formula for moments of a mixture

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt, nx1, nx2)

        """
        if j%2 == 0:
            return self.sig_v**j * special.factorial2(j-1)
        else:
            return np.zeros_like(self.sig_v)

    def __get_noncentral_moment_v__(self,
                                    axes_to_sum,
                                    z_in_conditioners,
                                    P_int,
                                    j,
                                    mu,
                                    light_weighted=False):
        """Helper function to get noncentral moments of p(v|...)

        Args:
            j (int): order of moment
            mu (array): center to take moment about, broadcastable to shape
                (nt,nx1,nx2)
            P_int (array): probability weights to integrate over, broadcastable
                to shape (nt,nx1,nx2)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            type: Description of returned object.

        """
        k = np.arange(0, j+1, 1)
        j_choose_k = special.comb(j, k)
        na = np.newaxis
        kw_get_p = {'light_weighted':light_weighted, 'density':False}
        if z_in_conditioners:
            E_v_txz = self.get_mean_v_txz(light_weighted=light_weighted)
            j_minus_k = np.broadcast_to(j-k, E_v_txz.shape+(j+1,))
            del_mu = E_v_txz - mu
            del_mu = del_mu[:,:,:,:,na]
            P_int = P_int[:,:,:,:,na]
            cent_moms = np.array([
                self.get_central_moment_v_txz(k0, light_weighted=light_weighted)
                for k0 in k])
        else:
            E_v_tx = self.get_mean_v_tx(light_weighted=light_weighted)
            j_minus_k = np.broadcast_to(j-k, E_v_tx.shape+(j+1,))
            del_mu = E_v_tx - mu
            del_mu = del_mu[:,:,:,na]
            P_int = P_int[:,:,:,na]
            cent_moms = np.array([
                self.get_central_moment_v_tx(k0, light_weighted=light_weighted)
                for k0 in k])
        cent_moms = np.moveaxis(cent_moms, 0, -1)
        integrand = del_mu**j_minus_k * cent_moms * P_int
        integral = np.sum(integrand, axes_to_sum)
        moment = np.sum(j_choose_k * integral, -1)
        return moment

    def get_noncentral_moment_v_tx(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|t,x) about arbitrary center mu

        int v (v - mu(t,x))^j p(v|t,x) dx

        Args:
            mu (array): arbitrary moment center, broadcastable with (nt,nx,ny)
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt, nx1, nx2)

        """
        dtx = self.cube.construct_volume_element('tx')
        P_int = np.ones(dtx.shape)
        moment = self.__get_noncentral_moment_v__(
            (),
            False,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_noncentral_moment_v_x(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|x) about arbitrary center mu

        int (v - mu(x))^j p(v|x) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with (nx,ny)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nx1, nx2)

        """
        P_int = self.get_p('t_x', light_weighted=light_weighted, density=False)
        moment = self.__get_noncentral_moment_v__(
            (0,),
            False,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v_x(self, j, light_weighted=False):
        """Get j'th central moment of p(v|x)

        int (v - E(v|x))^j p(v|x) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nx1, nx2)

        """
        E_v_x = self.get_mean_v_x(light_weighted=light_weighted)
        moment = self.get_noncentral_moment_v_x(
            j,
            E_v_x,
            light_weighted=light_weighted)
        return moment

    def get_noncentral_moment_v_t(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|t) about arbitrary center mu

        int (v - mu(t))^j p(v|t) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with (nt,)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,)

        """
        P_int = self.get_p('x_t', light_weighted=light_weighted, density=False)
        P_int = np.moveaxis(P_int, -1, 0)
        na = np.newaxis
        mu = mu[:,na,na]
        moment = self.__get_noncentral_moment_v__(
            (1,2),
            False,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v_t(self, j, light_weighted=False):
        """Get j'th central moment of p(v|t)

        int (v - E(v|t))^j p(v|t) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,)

        """
        E_v_t = self.get_mean_v_t(light_weighted=light_weighted)
        na = np.newaxis
        mom = self.get_noncentral_moment_v_t(
            j,
            E_v_t,
            light_weighted=light_weighted)
        return mom

    def get_noncentral_moment_v(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v) about arbitrary center mu

        int (v - mu)^j p(v) dv

        Args:
            j (int): order of moment
            mu (float): arbitrary moment center
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            float

        """
        P_int = self.get_p('tx', light_weighted=light_weighted, density=False)
        moment = self.__get_noncentral_moment_v__(
            (0,1,2),
            False,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v(self, j, mu, light_weighted=False):
        """Get j'th central moment of p(v)

        int (v - E(v))^j p(v) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            float

        """
        E_v = self.get_mean_v(light_weighted=light_weighted)
        moment = self.get_noncentral_moment_v(
            j,
            E_v,
            light_weighted=light_weighted)
        return moment

    def get_noncentral_moment_v_txz(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|t,x,z) about arbitrary center mu

        int (v - mu(t,x,z))^j p(v|t,x,z) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with
                (nt,nx1,nx2,nz)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,nx1,nx2,nz)

        """
        moment = self.get_noncentral_moment_v_tx(
            j,
            mu[:,:,:,0],    # E(v|t,x,z) = E(v|t,x,z0)
            light_weighted=light_weighted)
        nz = self.cube.ssps.delta_z.shape[0]
        moment = np.broadcast_to(moment, (nz,)+moment.shape)
        moment = np.moveaxis(moment, 0, -1)
        return moment

    def get_central_moment_v_txz(self, j, light_weighted=False):
        """Get j'th central moment of p(v|t,x,z)

        int (v - E(v|t,x,z))^j p(v|t,x,z) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with
                (nt,nx1,nx2,nz)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,nx1,nx2,nz)

        """
        moment = self.get_central_moment_v_tx(
            j,
            light_weighted=light_weighted)
        nz = self.cube.ssps.delta_z.shape[0]
        moment = np.broadcast_to(moment, (nz,)+moment.shape)
        moment = np.moveaxis(moment, 0, -1)
        return moment

    def get_noncentral_moment_v_tz(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|t,z) about arbitrary center mu

        int (v - mu(t,z))^j p(v|t,z) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with (nt,nz)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,nz)

        """
        P_int = self.get_p('x_tz', light_weighted=light_weighted, density=False)
        P_int = np.moveaxis(P_int, -2, 0)
        na = np.newaxis
        mu = mu[:,na,na,:]
        moment = self.__get_noncentral_moment_v__(
            (1,2),
            True,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v_tz(self, j, mu, light_weighted=False):
        """Get j'th central moment of p(v|t,z)

        int (v - E(v|t,z))^j p(v|t,z) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nt,nz)

        """
        E_v_tz = self.get_mean_v_tz(light_weighted=light_weighted)
        moment = self.get_noncentral_moment_v_tz(
            j,
            E_v_tz,
            light_weighted=light_weighted)
        return moment

    def get_noncentral_moment_v_xz(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|x,z) about arbitrary center mu

        int (v - mu(x,z))^j p(v|x,z) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with (nx1,nx2,nz)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nx1,nx2,nz)

        """
        P_int = self.get_p('t_xz', light_weighted=light_weighted, density=False)
        mu = mu[np.newaxis]
        moment = self.__get_noncentral_moment_v__(
            (0,),
            True,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v_xz(self, j, mu, light_weighted=False):
        """Get j'th central moment of p(v|x,z)

        int (v - E(v|x,z))^j p(v|x,z) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nx1,nx2,nz)

        """
        E_v_xz = self.get_mean_v_xz(light_weighted=light_weighted)
        moment = self.get_noncentral_moment_v_xz(
            j,
            E_v_xz,
            light_weighted=light_weighted)
        return moment

    def get_noncentral_moment_v_z(self, j, mu, light_weighted=False):
        """Get j'th noncentral moment of p(v|z) about arbitrary center mu

        int (v - mu(z))^j p(v|z) dv

        Args:
            j (int): order of moment
            mu (array): arbitrary moment center, broadcastable with (nz,)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nz,)

        """
        P_int = self.get_p('tx_z', light_weighted=light_weighted, density=False)
        na = np.newaxis
        mu = mu[na,na,na]
        moment = self.__get_noncentral_moment_v__(
            (0,1,2),
            True,
            P_int,
            j,
            mu,
            light_weighted=light_weighted)
        return moment

    def get_central_moment_v_z(self, j, mu, light_weighted=False):
        """Get j'th central moment of p(v|z)

        int (v - E(v|z))^j p(v|z) dv

        Args:
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array size (nz,)

        """
        E_v_z = self.get_mean_v_z(light_weighted=light_weighted)
        moment = self.get_noncentral_moment_v_z(
            j,
            E_v_z,
            light_weighted=light_weighted)
        return moment

    def get_mean_v_tx(self, light_weighted=False):
        """Get conditional expectation E(v|t,x)

        Returns:
            array size (nt, nx1, nx2)

        """
        return self.mu_v # for both light weighted and mass weighted

    def get_mean_v_x(self, light_weighted=False):
        """Get conditional expectation E(v|x)

        Returns:
            array size (nx1, nx2)

        """
        E_v_tx = self.get_mean_v_tx(light_weighted=light_weighted)
        P_t_x = self.get_p('t_x', light_weighted=light_weighted, density=False)
        E_v_x = np.sum(E_v_tx * P_t_x, 0)
        return E_v_x

    def get_mean_v_t(self, light_weighted=False):
        """Get conditional expectation E(v|t)

        Returns:
            array size (nt,)

        """
        E_v_tx = self.get_mean_v_tx(light_weighted=light_weighted)
        P_x_t = self.get_p('x_t', light_weighted=light_weighted, density=False)
        E_v_tx = np.moveaxis(E_v_tx, 0, -1)
        E_v_t = np.sum(E_v_tx * P_x_t, (0,1))
        return E_v_t

    def get_mean_v(self, light_weighted=False):
        """Get expectation E(v)

        Returns:
            float

        """
        E_v_tx = self.get_mean_v_tx(light_weighted=light_weighted)
        P_tx = self.get_p('tx', light_weighted=light_weighted, density=False)
        E_v = np.sum(E_v_tx*P_tx)
        return E_v

    def get_mean_v_txz(self, light_weighted=False):
        """Get conditional expectation E(v|t,x,z)

        Returns:
            array size (nt, nx1, nx2, nz)

        """
        E_v_tx = self.get_mean_v_tx(light_weighted=light_weighted)
        nz = self.cube.ssps.delta_z.shape[0]
        E_v_txz = np.broadcast_to(E_v_tx, (nz,)+E_v_tx.shape)
        E_v_txz = np.moveaxis(E_v_txz, 0, -1)
        return E_v_txz

    def get_mean_v_tz(self, light_weighted=False):
        """Get conditional expectation E(v|t,z)

        Returns:
            array size (nt, nz)

        """
        E_v_txz = self.get_mean_v_txz(light_weighted=light_weighted)
        P_x_tz = self.get_p('x_tz',
                            light_weighted=light_weighted,
                            density=False)
        na = np.newaxis
        E_v_txz = np.moveaxis(E_v_txz, 0, 2)
        E_v_tz = np.sum(E_v_txz*P_x_tz, (0,1))
        return E_v_tz

    def get_mean_v_xz(self, light_weighted=False):
        """Get conditional expectation E(v|x,z)

        Returns:
            array size (nx1, nx2, nz)

        """
        E_v_txz = self.get_mean_v_txz(light_weighted=light_weighted)
        P_t_xz = self.get_p('t_xz',
                            light_weighted=light_weighted,
                            density=False)
        E_v_xz = np.sum(E_v_txz*P_t_xz, 0)
        return E_v_xz

    def get_mean_v_z(self, light_weighted=False):
        """Get conditional expectation E(v|z)

        Returns:
            array size (nz,)

        """
        E_v_txz = self.get_mean_v_txz(light_weighted=light_weighted)
        P_tx_z = self.get_p('tx_z',
                            light_weighted=light_weighted,
                            density=False)
        E_v_z = np.sum(E_v_txz*P_tx_z, (0,1,2))
        return E_v_z