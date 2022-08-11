import numpy as np
from scipy import stats, special
from . import base

class stream(base.component):
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
        rotation: angle (radians) between x-axes of component and cube.
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
            theta_lims: (start, end) values of stream angle in radians. Must be
            in -pi to pi. To cross negative x-axis, set non-zero rotation when
            instantiating the stream component.
            mu_r_lims: (start, end) values of stream distance from center.
            sig (float): stream thickness.
            nsmp (int): number of points to sample the angle theta.

        Returns:
            type: Description of returned object.

        """
        assert np.min(theta_lims)>=-np.pi, "Angles must be in -pi<theta<pi'"
        assert np.max(theta_lims)<=np.pi, "Angles must be in -pi<theta<pi'"
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
        if self.theta_lims[0]<self.theta_lims[1]:
            mu_v_lo, mu_v_hi = mu_v_lims
        else:
            mu_v_hi, mu_v_lo = mu_v_lims
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


# end
