import numpy as np
from scipy import stats, special
from abc import ABC
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
    - the `get_p_vx` method, since generically p(v,x) = p(v|x)p(x)

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (degrees) between x-axes of component and cube.

    """

    def __init__(self,
                 cube=None,
                 center=(0,0),
                 rotation=0.):
        self.cube = cube
        self.center = center
        self.rotation = rotation
        theta = np.pi * rotation/180. # in radians
        costh = np.cos(theta)
        sinth = np.sin(theta)
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
        t_H = self.cube.ssps.par_edges[1][-1]
        log_t_dep_lim = np.log10(t_dep_lim)
        t_dep = np.logspace(*log_t_dep_lim, n_t_dep)
        z_lim = (z_max, 0., n_z)
        z = np.linspace(*z_lim, n_z)
        tt_dep, zz = np.meshgrid(t_dep, z, indexing='ij')
        t = t_H - tt_dep * zz/(1. - self.ahat * zz**self.bhat)
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
        p_v_x = self.get_p_v_x(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        p_x = self.get_p_x(density=density, light_weighted=light_weighted)
        p_vx = p_v_x*p_x
        return p_vx

class growingDisk(component):
    """A growing disk with age-and-space dependent velocities and enrichments

    The (mass-weighted) joint density of this component can be factorised as
    p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x)
    where the factors are given by:
    - p(t) : a beta distribution (see `set_p_t`)
    - p(x|t) : cored power-law stratified on elliptical radius with age-varying
    power law slope and x- and y- extents (see `set_p_x_t`)
    - p(v|t,x) : Gaussians with age-and-space varying means and dispersions.
    Means velocity maps resemble rotating disks (see `set_mu_v`) while
    dispersions drop off as power-laws on elliptical radii (see `set_sig_v`)
    - p(z|t,x) : chemical evolution model defined in equations 3-10 of
    Zhu et al 2020, parameterised by a spatially varying depletion
    timescale (see `set_p_z_tx`)

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (degrees) between x-axes of component and cube.

    """
    def __init__(self,
                 cube=None,
                 center=(0,0),
                 rotation=0.):
        super(growingDisk, self).__init__(
            cube=cube,
            center=center,
            rotation=rotation)

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

    def set_p_x_t(self,
                  sig_x_lims=(0.5, 0.5),
                  sig_y_lims=(0.5, 0.1),
                  alpha_lims=(0.5, 2.)):
        """Set the density p(x|t) as cored power-law in elliptical radius

        Desnities are cored power-laws stratified on elliptical radius r,
        r^2 = (x/sig_x)^2 + (y/sig_y)^2
        p(x|t) = (r+1)^-alpha
        where the disk sizes and slopes
        sig_x(t), sig_y(t), alpha(t)
        vary linearly with time between their specified (end, start) values.

        Args:
            sig_x_lims: (end,start) value of disk-size in x-direction
            sig_y_lims: (end,start) value of disk-size in y-direction
            alpha_lims: (end,start) value of power-law slope

        """
        # check input
        sig_x_lims = np.array(sig_x_lims)
        assert np.all(sig_x_lims > 0.)
        sig_y_lims = np.array(sig_y_lims)
        assert np.all(sig_y_lims > 0.)
        alpha_lims = np.array(alpha_lims)
        assert np.all(alpha_lims >= 0.)
        # get parameters vs time
        sig_xp = self.linear_interpolate_t(*sig_x_lims)
        sig_yp = self.linear_interpolate_t(*sig_y_lims)
        alpha = self.linear_interpolate_t(*alpha_lims)
        sig_xp = sig_xp[:, np.newaxis, np.newaxis]
        sig_yp = sig_yp[:, np.newaxis, np.newaxis]
        alpha = alpha[:, np.newaxis, np.newaxis]
        rr2 = (self.xxp/sig_xp)**2 + (self.yyp/sig_yp)**2
        rr = rr2 ** 0.5
        rho = (rr+1.) ** -alpha
        total_mass_per_t = np.sum(rho * self.cube.dx * self.cube.dy, (1,2))
        rho = (rho.T/total_mass_per_t).T
        self.x_t_pars = dict(sig_x_lims=sig_x_lims,
                             sig_y_lims=sig_y_lims,
                             alpha_lims=alpha_lims)
        # rearrange shape from [t,x,y] to match function signature [x,y,t]
        rho = np.rollaxis(rho, 0, 3)
        self.p_x_t = rho

    def set_t_dep(self,
                  sig_x=0.5,
                  sig_y=0.1,
                  alpha=1.5,
                  t_dep_in=0.5,
                  t_dep_out=6.):
        """Set spatially-varying depletion timescale

        t_dep varies as  power law in eliptical radius (with axes lengths
        `sig_x` and `sig_y`) with power-law slope `alpha`, from central value
        `t_dep_in` to outer value `t_dep_out`.

        Args:
            sig_x: x-axis length of ellipse of `t_dep` equicontours
            sig_y : y-axis length of ellipse of `t_dep` equicontours
            alpha : power law slope for varying `t_dep`
            t_dep_in : central value of `t_dep`
            t_dep_out : outer value of `t_dep`

        """
        # check input
        assert sig_x > 0.
        assert sig_y > 0.
        assert alpha >= 1.
        assert (t_dep_in > 0.1) and (t_dep_in < 10.0)
        assert (t_dep_out > 0.1) and (t_dep_out < 10.0)
        # evaluate t_dep maps
        rr2 = (self.xxp/sig_x)**2 + (self.yyp/sig_y)**2
        rr = rr2**0.5
        log_t_dep_in = np.log(t_dep_in)
        log_t_dep_out = np.log(t_dep_out)
        delta_log_t_dep = log_t_dep_in - log_t_dep_out
        log_t_dep = log_t_dep_out + delta_log_t_dep * alpha**-rr
        t_dep = np.exp(log_t_dep)
        self.t_dep_pars = dict(sig_x=sig_x,
                               sig_y=sig_y,
                               alpha=alpha,
                               t_dep_in=t_dep_in,
                               t_dep_out=t_dep_out)
        self.t_dep = t_dep

    def set_p_z_tx(self):
        """Set p(z|t,x) Zhu+20 given enrichment model and spatially varying t_dep

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

    def set_mu_v(self,
                 sig_x_lims=(0.5, 0.5),
                 sig_y_lims=(0.5, 0.1),
                 rmax_lims=(0.1, 1.),
                 vmax_lims=(50., 250.),
                 vinf_lims=(10., 50)):
        """Set age-and-space dependent mean velocities resembling rotating disks

        Mean velocity maps have rotation-curves along x-axis peaking at v_max at
        r_max then tending to vinf for larger r. Given by the equation:
        E[p(v|t,x)] = sgn(x) * |theta|/(pi/2) * (vinf + Kr/(r+1)^alpha)
        where
        x' = x/sig_x, y' = y/sig_y
        r^2 = (x')^2 + (y')^2,  theta = arctan(x'/y')
        K and alpha are chosen to give peak velocity vmax at distance rmax.
        The quantities sig_x, sig_y, rmax, vmax and vinf vary linearly with time
        between their specified (end, start) values.

        Args:
            sig_x_lims: (end,start) value of the x-extent of equicontours.
            sig_y_lims: (end,start) value of the y-extent of equicontours.
            rmax_lims: distance of maximum velocity along x-axis.
            vmax_lims: maximum velocity along x-axis.
            vinf_lims: limit of velocity at large distance along x-axis.

        """
        # check input
        sig_x_lims = np.array(sig_x_lims)
        assert np.all(sig_x_lims > 0.)
        sig_y_lims = np.array(sig_y_lims)
        assert np.all(sig_y_lims > 0.)
        #alpha_lims = np.array(alpha_lims)
        #assert np.all(alpha_lims >= 1.)
        vmax_lims = np.array(vmax_lims)
        vinf_lims = np.array(vinf_lims)
        sign_vmax = np.sign(vmax_lims)
        sign_vinf = np.sign(vinf_lims)
        # check vmax's and v0's have consistent directions and magnitudes
        all_positive = np.isin(sign_vmax, [0,1])
        all_negative = np.isin(sign_vmax, [0,-1])
        assert np.all(all_positive) or np.all(all_negative)
        all_positive = np.isin(sign_vinf, [0,1])
        all_negative = np.isin(sign_vinf, [0,-1])
        assert np.all(all_positive) or np.all(all_negative)
        assert np.all(np.abs(vmax_lims) >= np.abs(vinf_lims))
        # linearly interpolate and reshape inputs
        sig_xp = self.linear_interpolate_t(*sig_x_lims)
        sig_yp = self.linear_interpolate_t(*sig_y_lims)
        rmax = self.linear_interpolate_t(*rmax_lims)
        # next three lines calculate alpha_lims just to add to `self.mu_v_pars`
        # to be able to compare `alpha_lims` with earlier implementations
        rmax_lims = np.array(rmax_lims)
        sig_x_lims = np.array(sig_x_lims)
        alpha_lims = (rmax_lims/sig_x_lims+1.)/(rmax_lims/sig_x_lims)
        alpha = (rmax/sig_xp+1.)/(rmax/sig_xp)
        vmax = self.linear_interpolate_t(*vmax_lims)
        vinf = self.linear_interpolate_t(*vinf_lims)
        sig_xp = sig_xp[:, np.newaxis, np.newaxis]
        sig_yp = sig_yp[:, np.newaxis, np.newaxis]
        alpha = alpha[:, np.newaxis, np.newaxis]
        vmax = vmax[:, np.newaxis, np.newaxis]
        vinf = vinf[:, np.newaxis, np.newaxis]
        # make mu_v maps
        th = np.arctan(self.xxp/sig_xp/self.yyp*sig_yp)
        idx = np.where(self.yyp==0)
        th[:, idx[0], idx[1]] = np.pi/2.
        th = np.abs(th)
        th = th/np.pi*2.
        self.th = th
        rr2 = (self.xxp/sig_xp)**2 + (self.yyp/sig_yp)**2
        rr = rr2**0.5
        alpha_m1 = alpha - 1.
        k = (vmax-vinf) * alpha**alpha / alpha_m1**alpha_m1
        mu_v = vinf + k * rr / (rr+1.)**alpha
        mu_v *= th
        mu_v *= np.sign(self.xxp)
        self.mu_v_pars = dict(sig_x_lims=sig_x_lims,
                              sig_y_lims=sig_y_lims,
                              rmax_lims=rmax_lims,
                              vmax_lims=vmax_lims,
                              vinf_lims=vinf_lims,
                              alpha_lims=alpha_lims)
        self.mu_v = mu_v

    def set_sig_v(self,
                  sig_x_lims=(0.5, 0.5),
                  sig_y_lims=(0.5, 0.1),
                  alpha_lims=(1.5, 2.5),
                  sig_v_in_lims=(50., 250.),
                  sig_v_out_lims=(10., 50)):
        """Set age-and-space dependent velocity dispersion maps

        Dispersion maps vary as power-laws between central value sig_v_in, outer
        value sig_v_out, with slopes alpha. Velocity dispersion is constant on
        ellipses with axis-lengths sig_x and sig_y. The quantities sig_x, sig_y,
        alpha, sig_v_in, sig_v_out vary linearly with time between their
        specified (end, start) values.

        Args:
            sig_x_lims: (end,start) value of the x-extent of equicontours.
            sig_y_lims: (end,start) value of the y-extent of equicontours.
            alpha_lims: (end,start) value of power-law slope.
            sig_v_in_lims: (end,start) value of central dispersion.
            sig_v_out_lims: (end,start) value of outer dispersion.

        """
        # check input
        sig_x_lims = np.array(sig_x_lims)
        assert np.all(sig_x_lims > 0.)
        sig_y_lims = np.array(sig_y_lims)
        assert np.all(sig_y_lims > 0.)
        alpha_lims = np.array(alpha_lims)
        assert np.all(alpha_lims >= 1.)
        sig_v_in_lims = np.array(sig_v_in_lims)
        assert np.all(sig_v_in_lims > 0.)
        sig_v_out_lims = np.array(sig_v_out_lims)
        assert np.all(sig_v_out_lims > 0.)
        # linearly interpolate and reshape inputs
        sig_xp = self.linear_interpolate_t(*sig_x_lims)
        sig_yp = self.linear_interpolate_t(*sig_y_lims)
        alpha = self.linear_interpolate_t(*alpha_lims)
        sig_v_in = self.linear_interpolate_t(*sig_v_in_lims)
        sig_v_out = self.linear_interpolate_t(*sig_v_out_lims)
        sig_xp = sig_xp[:, np.newaxis, np.newaxis]
        sig_yp = sig_yp[:, np.newaxis, np.newaxis]
        alpha = alpha[:, np.newaxis, np.newaxis]
        sig_v_in = sig_v_in[:, np.newaxis, np.newaxis]
        sig_v_out = sig_v_out[:, np.newaxis, np.newaxis]
        # evaluate sig_v maps
        rr2 = (self.xxp/sig_xp)**2 + (self.yyp/sig_yp)**2
        rr = rr2**0.5
        log_sig_v_in = np.log(sig_v_in)
        log_sig_v_out = np.log(sig_v_out)
        delta_log_sig_v = log_sig_v_in - log_sig_v_out
        log_sig = log_sig_v_out + delta_log_sig_v * alpha**-rr
        sig = np.exp(log_sig)
        self.sig_v_pars = dict(sig_x_lims=sig_x_lims,
                               sig_y_lims=sig_y_lims,
                               alpha_lims=alpha_lims,
                               sig_v_in_lims=sig_v_in_lims,
                               sig_v_out_lims=sig_v_out_lims)
        self.sig_v = sig

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
        p_x_t = self.get_p_x_t(density=density, light_weighted=light_weighted)
        p_t = self.get_p_t(density=density, light_weighted=light_weighted)
        p_xt = p_x_t*p_t
        p_tx = np.einsum('xyt->txy', p_xt)
        return p_tx

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
        new_ax = np.newaxis
        p_t = self.get_p_t(density=density)
        p_t = p_t[:, new_ax, new_ax, new_ax]
        p_x_t = self.get_p_x_t(density=density)
        p_x_t = np.rollaxis(p_x_t, 2, 0)
        p_x_t = p_x_t[:, :, :, new_ax]
        p_z_tx = self.get_p_z_tx(density=density)
        p_z_tx = np.rollaxis(p_z_tx, 0, 4)
        p_txz = p_t * p_x_t * p_z_tx
        if light_weighted:
            ssps = self.cube.ssps
            light_weights = ssps.light_weights[:,new_ax,new_ax,:]
            P_txz_mass_wtd = self.get_p_txz(density=False)
            normalisation = np.sum(P_txz_mass_wtd*light_weights)
            p_txz = p_txz*light_weights/normalisation
        return p_txz

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

    def get_p_tz_x(self, density=True, light_weighted=False):
        """Get p(t,z|x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        new_ax = np.newaxis
        # evaluate both as densities...
        p_txz = self.get_p_txz(density=density)
        p_x = self.get_p_x(density=density)
        # ... since dx appears on top and bottom, hence cancel
        p_tz_x = p_txz/p_x[new_ax,:,:,new_ax]   # shape txz
        p_tz_x = np.rollaxis(p_tz_x, 3, 1)      # shape tzx
        if light_weighted:
            ssps = self.cube.ssps
            light_weights = ssps.light_weights[:,:,new_ax,new_ax]
            P_tz_x_mass_wtd = self.get_p_tz_x(density=False)
            normalisation = np.sum(P_tz_x_mass_wtd*light_weights, (0,1))
            p_tz_x = p_tz_x*light_weights/normalisation
        return p_tz_x

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
        p_tz_x = self.get_p_tz_x(density=density)
        P_x = self.get_p_x(density=False)
        p_tz = np.sum(p_tz_x * P_x, (2,3))
        if light_weighted:
            ssps = self.cube.ssps
            P_tz_mass_wtd = self.get_p_tz(density=False)
            normalisation = np.sum(P_tz_mass_wtd*ssps.light_weights)
            p_tz = p_tz*ssps.light_weights/normalisation
        return p_tz

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
        na = np.newaxis
        if light_weighted is False:
            v_edg = v_edg[:, na, na, na]
            norm = stats.norm(loc=self.mu_v, scale=self.sig_v)
            p_v_tx = norm.cdf(v_edg[1:]) - norm.cdf(v_edg[:-1])
            if density is True:
                dv = v_edg[1:] - v_edg[:-1]
                p_v_tx /= dv
        else:
            p_tvxz = self.get_p_tvxz(v_edg, density=True, light_weighted=True)
            if density is False:
                dv = v_edg[1:] - v_edg[:-1]
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
        p_txz = self.get_p_txz(density=density)
        p_v_tx = self.get_p_v_tx(v_edg, density=density)
        newax = np.newaxis
        p_v_txz = p_v_tx[:, :, :, :, newax]
        p_txz = p_txz[newax, :, :, :, :]
        p_vtxz = p_v_txz * p_txz
        p_tvxz = np.einsum('vtxyz->tvxyz', p_vtxz)
        if light_weighted:
            ssps = self.cube.ssps
            light_weights = ssps.light_weights
            light_weights = light_weights[:,newax,newax,newax,:]
            P_tvxz_mass_wtd = self.get_p_tvxz(v_edg, density=False)
            normalisation = np.sum(P_tvxz_mass_wtd*light_weights)
            p_tvxz = p_tvxz*light_weights/normalisation
        return p_tvxz

    def get_p_v_x(self, v_edg, density=True, light_weighted=False):
        """Get p(v|x)

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
        p_v_tx = self.get_p_v_tx(v_edg=v_edg,
                                 density=density,
                                 light_weighted=light_weighted)
        P_t = self.get_p_t(density=False, light_weighted=light_weighted)
        P_t = P_t[na, :, na, na]
        p_v_x = np.sum(p_v_tx * P_t, 1)
        return p_v_x

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
        p_v_x = self.get_p_v_x(v_edg,
                               density=density,
                               light_weighted=light_weighted)
        P_x = self.get_p_x(density=False, light_weighted=light_weighted)
        p_v = np.sum(p_v_x*P_x, (1,2))
        return p_v

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

    def get_variance_v_x(self, light_weighted=False):
        """Get variance velocity map

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        var_v_x = self.get_jth_central_moment_v_x(
            2,
            light_weighted=light_weighted)
        return var_v_x

    def get_skewness_v_x(self, light_weighted=False):
        """Get skewness of velocity map

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        mu3_v_x = self.get_jth_central_moment_v_x(
            3,
            light_weighted=light_weighted)
        var_v_x = self.get_jth_central_moment_v_x(
            2,
            light_weighted=light_weighted)
        skewness_v_x = mu3_v_x/var_v_x**1.5
        return skewness_v_x

    def get_kurtosis_v_x(self, light_weighted=False):
        """Get kurtosis of velocity map

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        mu4_v_x = self.get_jth_central_moment_v_x(
            4,
            light_weighted=light_weighted)
        var_v_x = self.get_jth_central_moment_v_x(
            2,
            light_weighted=light_weighted)
        kurtosis_v_x = mu4_v_x/var_v_x**2.
        return kurtosis_v_x

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
                              colorbar_label='$t_\mathrm{dep}$',
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


class stream(component):
    """A stream with spatially varying kinematics but uniform enrichment.

    The (mass-weighted) joint density of this component can be factorised as
    p(t,x,v,z) = p(t) p(x) p(v|x) p(z|t)
    where the factors are given by:
    - p(t) : a beta distribution (see `set_p_t`),
    - p(x) : a curved line with constant thickness (see `set_p_x`),
    - p(v|x) : Guassian with mean varying along stream and constant sigma (see
    set_p_v_x`),
    - p(z|t) : single chemical evolution track i.. `t_dep` (see `set_p_z_t`).

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (degrees) between x-axes of component and cube.
        nsmp:

    """
    def __init__(self,
                 cube=None,
                 center=(0,0),
                 rotation=0.):
        super(stream, self).__init__(
            cube=cube,
            center=center,
            rotation=rotation)

    def set_p_x(self,
                theta_lims=[0., np.pi/2.],
                mu_r_lims=[0.7, 0.1],
                sig=0.03,
                nsmp=75):
        """Define the stream track p(x)

        Defined in polar co-ordinates (theta,r). Stream extends between angles
        `theta_lims` between radii in `mu_r_lims`. Density is constant along
        with varying theta. The track has a constant width on the sky, `sig`.

        Args:
            theta_lims: (start, end) values of stream angle in radians.
            mu_r_lims: (start, end) values of stream distance from center.
            sig (float): stream thickness.
            nsmp (int): number of points to sample the angle theta.

        Returns:
            type: Description of returned object.

        """
        self.theta_lims = theta_lims
        cube = self.cube
        theta0, theta1 = theta_lims
        self.nsmp = nsmp
        theta_smp = np.linspace(theta0, theta1, self.nsmp)
        mu_r0, mu_r1 = mu_r_lims
        tmp = (theta_smp - theta0)/(theta1 - theta0)
        mu_r_smp =  mu_r0 + (mu_r1 - mu_r0) * tmp
        mu_x_smp = mu_r_smp * np.cos(theta_smp)
        nrm_x = stats.norm(mu_x_smp, sig)
        pdf_x = nrm_x.cdf(self.xxp[:,:,np.newaxis] + cube.dx/2.)
        pdf_x -= nrm_x.cdf(self.xxp[:,:,np.newaxis] - cube.dx/2.)
        mu_y_smp = mu_r_smp * np.sin(theta_smp)
        nrm_y = stats.norm(mu_y_smp, sig)
        pdf_y = nrm_y.cdf(self.yyp[:,:,np.newaxis] + cube.dy/2.)
        pdf_y -= nrm_y.cdf(self.yyp[:,:,np.newaxis] - cube.dy/2.)
        pdf = pdf_x * pdf_y
        pdf = np.sum(pdf, -1)
        pdf /= np.sum(pdf*cube.dx*cube.dy)
        self.p_x_pars = dict(theta_lims=theta_lims,
                             mu_r_lims=mu_r_lims,
                             sig=sig,
                             nsmp=nsmp)
        self.p_x = pdf

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
        # since x is indpt of (t,z), light- and mass- weight densities are equal
        p_x = self.p_x.copy()
        if density is False:
            p_x *= (self.cube.dx * self.cube.dy)
        return p_x

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
        p_x = self.get_p_x(density=density, light_weighted=light_weighted)
        # since x is indpt of t, p(x|t)=p(x)
        nt = self.cube.ssps.par_dims[1]
        p_x_t = np.broadcast_to(p_x[:,:,np.newaxis], p_x.shape+(nt,))
        return p_x_t

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
        p_x = self.get_p_x(density=density, light_weighted=light_weighted)
        p_t = self.get_p_t(density=density, light_weighted=light_weighted)
        # since x and t are independent p(t,x)=p(x)p(t)
        na = np.newaxis
        p_tx = p_t[:,na,na]*p_x[na,:,:]
        return p_tx

    def set_p_z_t(self, t_dep=3.):
        assert (t_dep > 0.1) and (t_dep < 10.0)
        self.t_dep = t_dep
        del_t_dep = self.t_dep - self.z_t_interp_grid['t_dep']
        abs_del_t_dep = np.abs(del_t_dep)
        idx_t_dep = np.argmin(abs_del_t_dep, axis=-1)
        self.idx_t_dep = idx_t_dep
        idx_t_dep = np.ravel(idx_t_dep)
        z_mu = self.z_t_interp_grid['z'][idx_t_dep]
        z_mu = np.squeeze(z_mu)
        self.z_mu = z_mu
        z_sig2 = self.ahat * z_mu**self.bhat
        log_z_edg = self.cube.ssps.par_edges[0]
        del_log_z = log_z_edg[1:] - log_z_edg[:-1]
        x_xsun = 1. # i.e. assuming galaxy has hydrogen mass fraction = solar
        lin_z_edg = 10.**log_z_edg * x_xsun
        nrm = stats.norm(loc=z_mu, scale=z_sig2**0.5)
        lin_z_edg = lin_z_edg[:, np.newaxis]
        cdf_z_tx = nrm.cdf(lin_z_edg)
        p_z_t = cdf_z_tx[1:] - cdf_z_tx[:-1]
        p_z_t /= np.sum(p_z_t, 0)
        p_z_t /= del_log_z[:, np.newaxis]
        self.p_z_t_pars = dict(t_dep=t_dep)
        self.p_z_t = p_z_t

    def get_p_z_t(self, density=True, light_weighted=False):
        """Get p(z|t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        if light_weighted is False:
            p_z_t = self.p_z_t.copy()
            if density is False:
                dz = self.cube.ssps.delta_z
                dz = dz[:, np.newaxis]
                p_z_t *= dz
        else:
            p_tz = self.get_p_tz(density=False, light_weighted=True)
            p_t = self.get_p_t(density=False, light_weighted=True)
            p_z_t = p_tz.T/p_t
            if density:
                dz = self.cube.ssps.delta_z
                dz = dz[:, np.newaxis]
                p_z_t /= dz
        return p_z_t

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
        p_z_t = self.get_p_z_t(density=density, light_weighted=False)
        p_t = self.get_p_t(density=density, light_weighted=False)
        p_zt = p_z_t*p_t
        p_tz = p_zt.T
        if light_weighted:
            ssps = self.cube.ssps
            P_tz_mass_wtd = self.get_p_tz(density=False)
            normalisation = np.sum(P_tz_mass_wtd*ssps.light_weights)
            p_tz = p_tz*ssps.light_weights/normalisation
        return p_tz

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
        p_tz = self.get_p_tz(density=False, light_weighted=light_weighted)
        p_z = np.sum(p_tz, 0)
        if density is True:
            dz = self.cube.ssps.delta_z
            p_z /= dz
        return p_z

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
        p_z_t = self.get_p_z_t(density=density, light_weighted=light_weighted)
        p_z_tx = p_z_t[:,:,np.newaxis,np.newaxis]
        cube_shape = (self.cube.nx, self.cube.ny)
        p_z_tx = np.broadcast_to(p_z_tx, p_z_t.shape+cube_shape)
        return p_z_tx

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
        p_tz = self.get_p_tz(density=density, light_weighted=light_weighted)
        p_x = self.get_p_x(density=density, light_weighted=light_weighted)
        na = np.newaxis
        p_txz = p_tz[:,na,na,:] * p_x[na,:,:,na]
        return p_txz

    def get_p_tz_x(self, density=True, light_weighted=False):
        """Get p(t,z|x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tz = self.get_p_tz(density=density, light_weighted=light_weighted)
        na = np.newaxis
        cube_shape = (self.cube.nx, self.cube.ny)
        p_tz_x = np.broadcast_to(p_tz[:,:,na,na], p_tz.shape+cube_shape)
        return p_tz_x

    def set_p_v_x(self,
                  mu_v_lims=[-100,100],
                  sig_v=100.):
        """Set parameters for p(v|x)

        p(v|x) = N(mu_v(x), sig) where mu_v(x) is linearly interpolated with
        angle theta, between start/end values specified in `mu_v_lims`

        Args:
            mu_v_lims: (start, end) values of stream velocity
            sig_v (float): constant std dev of velocity distribution

        """
        th = np.arctan2(self.yyp, self.xxp)
        mu_v = np.zeros_like(th)
        mu_v_lo, mu_v_hi = mu_v_lims
        min_th, max_th = np.min(self.theta_lims), np.max(self.theta_lims)
        idx = np.where(th <= min_th)
        mu_v[idx] = mu_v_lo
        idx = np.where(th >= max_th)
        mu_v[idx] = mu_v_hi
        idx = np.where((th > min_th) & (th < max_th))
        mu_v[idx] = (th[idx]-min_th)/(max_th-min_th) * (mu_v_hi-mu_v_lo)
        mu_v[idx] += mu_v_lo
        self.p_v_x_pars = dict(mu_v_lims=mu_v_lims, sig_v=sig_v)
        self.mu_v = mu_v
        self.sig_v = np.zeros_like(mu_v) + sig_v

    def get_p_v_x(self, v_edg, density=True, light_weighted=False):
        """Get p(v|x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        norm = stats.norm(loc=self.mu_v, scale=self.sig_v)
        v = (v_edg[:-1] + v_edg[1:])/2.
        v = v[:, np.newaxis, np.newaxis]
        p_v_x = norm.pdf(v)
        if density is False:
            dv = v_edg[1:] - v_edg[:-1]
            dv = dv[:, np.newaxis, np.newaxis]
            p_v_x *= dv
        return p_v_x

    def get_p_v(self, v_edg, density=True, light_weighted=False):
        """Get p(v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_v_x = self.get_p_v_x(v_edg, density=density)
        P_x = self.get_p_x(density=False)
        p_v = np.sum(p_v_x*P_x, (1,2))
        return p_v

    def get_p_v_tx(self, v_edg, density=True, light_weighted=False):
        """Get p(v|t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_v_x = self.get_p_v_x(v_edg, density=density)
        nt = self.cube.ssps.par_dims[1]
        shape = p_v_x.shape
        shape = (shape[0], nt, shape[1], shape[2])
        p_v_tx = np.broadcast_to(p_v_x[:,np.newaxis,:,:], shape)
        return p_v_tx

    def get_p_tvxz(self, v_edg, density=True, light_weighted=False):
        """Get p(v,t,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        p_tz = self.get_p_tz(density=density, light_weighted=light_weighted)
        p_v_x = self.get_p_v_x(v_edg, density=density)
        p_x = self.get_p_x(density=density)
        na = np.newaxis
        p_vx = p_v_x * p_x[na,:,:]
        p_tvxz = p_tz[:,na,na,na,:] * p_vx[na,:,:,:,na]
        return p_tvxz


# end
