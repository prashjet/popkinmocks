import numpy as np
from scipy import stats
from . import parametric

class GrowingDisk(parametric.ParametricComponent):
    """A growing disk with age-and-space dependent velocities and enrichments

    The (mass-weighted) joint density of this component can be factorised as
    p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x)
    where the factors are given by:
    - p(t) : a beta distribution (see `set_p_t`)
    - p(x|t) : cored power-law in elliptical radius with age-varying core-size,
    flattening and slope (see `set_p_x_t`)
    - p(v|t,x) = Normal(v ; mu_v(t,x), sig_v(t,x)) where mean maps resemble
    rotating disks (see `set_mu_v`) while dispersion varies as power-laws on
    elliptical radius (see `set_sig_v`)
    - p(z|t,x) = Normal(z ; mu_z(t, t_dep(x)), sig_z(t, t_dep(x))) i.e. chemical
    enrichment (i.e. metallicity vs t) depends on depletion timescale t_dep(x)
    varying as as power law in eliptical radius (see `set_t_dep`). The functions
    mu_z(t,t_dep) and sig_z(t,t_dep) are taken from equations 3-10 of Zhu, van
    de Venn, Leaman et al 20.

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.

    """

    def set_p_x_t(self,
                  q_lims=(0.5, 0.1),
                  rc_lims=(0.5, 0.1),
                  alpha_lims=(0.5, 2.)):
        """Set the density p(x|t) as cored power-law in elliptical radius

        Desnities are cored power-laws stratified on elliptical radius r,
        r^2 = x^2 + (y/q)^2
        p(x|t) = (r+rc)^-alpha
        where the disk axis ratio q(t), slope alpha(t) and core radius rc(t)
        vary linearly with t between values specified for (young, old) stars.

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

        t_dep varies as power law in eliptical radius with axis ratio q with
        power-law slope alpha, from central value t_dep_in to outer
        value t_dep_out.

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
        The quantities q, rmax and vmax vary linearly with t between the values
        specified for (young, old) stars.

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
