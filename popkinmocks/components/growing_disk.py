import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from . import base

class growingDisk(base.component):
    """A growing disk with age-and-space dependent velocities and enrichments

    The (mass-weighted) joint density of this component can be factorised as
    p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x)
    where the factors are given by:
    - p(t) : a beta distribution (see `set_p_t`)
    - p(x|t) : cored power-law stratified with age-varying flattening and slope
    (see `set_p_x_t`)
    - p(v|t,x) : Gaussians with age-and-space varying means and dispersions.
    Means velocity maps resemble rotating disks (see `set_mu_v`) while
    dispersions drop off as power-laws on ellipses (see `set_sig_v`)
    - p(z|t,x) : chemical evolution model defined in equations 3-10 of
    Zhu et al 2020, parameterised by a spatially varying depletion
    timescale (see `set_p_z_tx`)

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.

    """
    def __init__(self,
                 cube=None,
                 center=(0,0),
                 rotation=0.):
        super(growingDisk, self).__init__(
            cube=cube,
            center=center,
            rotation=rotation)

    def set_p_x_t(self,
                  q_lims=(0.5, 0.1),
                  rc_lims=(0.5, 0.1),
                  alpha_lims=(0.5, 2.)):
        """Set the density p(x|t) as cored power-law in elliptical radius

        Desnities are cored power-laws stratified on elliptical radius r,
        r^2 = x^2 + (y/q)^2
        p(x|t) = (r+rc)^-alpha
        where the disk axis ratio q(t) and slope alpha(t) vary linearly with
        stellar age between values specified for (young, old) stars.

        Args:
            q_lims: (young,old) value of disk y/x axis ratio
            rc_lims: (young,old) value of disk core-size in elliptical radii
            alpha_lims: (young,old) value of power-law slope

        """
        # check input
        q_lims = np.array(q_lims)
        assert np.all(q_lims > 0.)
        rc_lims = np.array(rc_lims)
        assert np.all(rc_lims > 0.)
        alpha_lims = np.array(alpha_lims)
        assert np.all(alpha_lims >= 0.)
        # get parameters vs time
        q = self.linear_interpolate_t(*q_lims)
        rc = self.linear_interpolate_t(*rc_lims)
        alpha = self.linear_interpolate_t(*alpha_lims)
        q = q[:, np.newaxis, np.newaxis]
        rc = rc[:, np.newaxis, np.newaxis]
        alpha = alpha[:, np.newaxis, np.newaxis]
        rr2 = self.xxp**2 + (self.yyp/q)**2
        rr = rr2 ** 0.5
        rho = (rr+rc) ** -alpha
        total_mass_per_t = np.sum(rho * self.cube.dx * self.cube.dy, (1,2))
        rho = (rho.T/total_mass_per_t).T
        self.x_t_pars = dict(q_lims=q_lims,
                             rc_lims=rc_lims,
                             alpha_lims=alpha_lims)
        # rearrange shape from [t,x,y] to match function signature [x,y,t]
        rho = np.rollaxis(rho, 0, 3)
        self.p_x_t = rho

    def set_t_dep(self,
                  q=0.1,
                  alpha=1.5,
                  t_dep_in=0.5,
                  t_dep_out=6.):
        """Set spatially-varying depletion timescale

        t_dep varies as  power law in eliptical radius (with axis ratio `q`)
        with power-law slope `alpha`, from central value `t_dep_in` to outer
        value `t_dep_out`.

        Args:
            q : y/x axis ratio of ellipses of `t_dep` equicontours
            alpha : power law slope for varying `t_dep`
            t_dep_in : central value of `t_dep`
            t_dep_out : outer value of `t_dep`

        """
        # check input
        assert q > 0.
        assert alpha >= 1.
        assert (t_dep_in > 0.1) and (t_dep_in < 10.0)
        assert (t_dep_out > 0.1) and (t_dep_out < 10.0)
        # evaluate t_dep maps
        rr2 = self.xxp**2 + (self.yyp/q)**2
        rr = rr2**0.5
        log_t_dep_in = np.log(t_dep_in)
        log_t_dep_out = np.log(t_dep_out)
        delta_log_t_dep = log_t_dep_in - log_t_dep_out
        log_t_dep = log_t_dep_out + delta_log_t_dep * alpha**-rr
        t_dep = np.exp(log_t_dep)
        self.t_dep_pars = dict(q=q,
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
                 q_lims=(0.5, 0.5),
                 rmax_lims=(0.1, 1.),
                 vmax_lims=(50., 250.)):
        """Set age-and-space dependent mean velocities resembling rotating disks

        Mean velocity maps have rotation-curves along x-axis peaking at v_max at
        r_max then falling to 0 for r->inf. Given by the equation:
        E[p(v|t,[x,y])] = cos(theta) * Kr/(r+rc)^3
        where
        r^2 = x^2 + (y/q)^2,  theta = arctan(x/(y/q))
        K and rc are chosen to give peak velocity vmax at distance rmax.
        The quantities q, rmax and vmax vary linearly with stellar age between
        the values specified for (young,old) stars.

        Args:
            q_lims: (young,old) value of y/x axis ratio of mu(v) equicontours.
            rmax_lims: (young,old) distance of maximum velocity along x-axis.
            vmax_lims: (young,old) maximum velocity.

        """
        # check input
        q_lims = np.array(q_lims)
        assert np.all(q_lims > 0.)
        rmax_lims = np.array(rmax_lims)
        vmax_lims = np.array(vmax_lims)
        sign_vmax = np.sign(vmax_lims)
        # check vmax's have consistent directions and magnitudes
        all_positive = np.isin(sign_vmax, [0,1])
        all_negative = np.isin(sign_vmax, [0,-1])
        assert np.all(all_positive) or np.all(all_negative)
        # linearly interpolate and reshape inputs
        q = self.linear_interpolate_t(*q_lims)
        rmax = self.linear_interpolate_t(*rmax_lims)
        vmax = self.linear_interpolate_t(*vmax_lims)
        rc = 2.*rmax
        K = vmax*(rmax+rc)**3/rmax
        q = q[:, np.newaxis, np.newaxis]
        rc = rc[:, np.newaxis, np.newaxis]
        K = K[:, np.newaxis, np.newaxis]
        # make mu_v maps
        th = np.arctan2(self.yyp/q, self.xxp)
        # idx = np.where(self.xxp==0)
        # th[:, idx[0], idx[1]] = np.pi/2.
        rr2 = self.xxp**2 + (self.yyp/q)**2
        rr = rr2**0.5
        mu_v = K*rr/(rr+rc)**3 * np.cos(th)
        self.mu_v_pars = dict(q_lims=q_lims,
                              rmax_lims=rmax_lims,
                              vmax_lims=vmax_lims)
        self.mu_v = mu_v

    def set_sig_v(self,
                  q_lims=(0.5, 0.1),
                  alpha_lims=(1.5, 2.5),
                  sig_v_in_lims=(50., 250.),
                  sig_v_out_lims=(10., 50)):
        """Set age-and-space dependent velocity dispersion maps

        Dispersion maps vary as power-laws between central value sig_v_in, outer
        value sig_v_out, with slopes alpha. Velocity dispersion is constant on
        ellipses with y/x axis-ratio = q. The quantities q, alpha, sig_v_in,
        sig_v_out vary linearly with stellar age between values specified for
        (young,old) stars.

        Args:
            q_lims: (young,old) values of y/x axis-ratio of sigma equicontours.
            alpha_lims: (young,old) value of power-law slope.
            sig_v_in_lims: (young,old) value of central dispersion.
            sig_v_out_lims: (young,old) value of outer dispersion.

        """
        # check input
        q_lims = np.array(q_lims)
        assert np.all(q_lims > 0.)
        alpha_lims = np.array(alpha_lims)
        assert np.all(alpha_lims >= 1.)
        sig_v_in_lims = np.array(sig_v_in_lims)
        assert np.all(sig_v_in_lims > 0.)
        sig_v_out_lims = np.array(sig_v_out_lims)
        assert np.all(sig_v_out_lims > 0.)
        # linearly interpolate and reshape inputs
        q = self.linear_interpolate_t(*q_lims)
        alpha = self.linear_interpolate_t(*alpha_lims)
        sig_v_in = self.linear_interpolate_t(*sig_v_in_lims)
        sig_v_out = self.linear_interpolate_t(*sig_v_out_lims)
        q = q[:, np.newaxis, np.newaxis]
        alpha = alpha[:, np.newaxis, np.newaxis]
        sig_v_in = sig_v_in[:, np.newaxis, np.newaxis]
        sig_v_out = sig_v_out[:, np.newaxis, np.newaxis]
        # evaluate sig_v maps
        rr2 = self.xxp**2 + (self.yyp/q)**2
        rr = rr2**0.5
        log_sig_v_in = np.log(sig_v_in)
        log_sig_v_out = np.log(sig_v_out)
        delta_log_sig_v = log_sig_v_in - log_sig_v_out
        log_sig = log_sig_v_out + delta_log_sig_v * alpha**-rr
        sig = np.exp(log_sig)
        self.sig_v_pars = dict(q_lims=q_lims,
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
