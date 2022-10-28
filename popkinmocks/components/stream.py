import numpy as np
from scipy import stats, special
from . import parametric

class Stream(parametric.ParametricComponent):
    """Stream with age-independent kinematics and spatially-uniform enrichment

    The mass-weighted density factorises as
    p(t,x,v,z) = p(t) p(x|t) p(v|t,x) p(z|t,x), where:
    - p(t) : beta distribution (see `set_p_t`),
    - p(x|t) = p(x): a curved line with constant thickness (see `set_p_x`),
    - p(v|t,x) = p(v|x) = Normal(v ; mu_v(x), sig_v(x)) where mean varies along
    stream (see `set_mu_v`) and dispersion is constant (see `set_sig_v`)
    - p(z|t,x) = p(z|t) = Normal(z ; mu_z(t, t_dep), sig_z(t, t_dep)) i.e.
    chemical enrichment depends on a constant depletion timescale t_dep (see
    `set_t_dep`). The functions mu_z(t,t_dep) and sig_z(t,t_dep) are from
    equations 3-10 of Zhu et al. 20,
    https://ui.adsabs.harvard.edu/abs/2020MNRAS.496.1579Z/abstract

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.

    """
    def set_p_x_t(self,
                  theta_lims=[0., np.pi/2.],
                  mu_r_lims=[0.7, 0.1],
                  sig=0.03,
                  nsmp=75):
        """Define the stream track p(x)

        Defined in polar co-ordinates (theta,r). Stream extends between angles
        `theta_lims` between radii in `mu_r_lims`. Density is constant with
        varying theta. The track has a constant width on the sky, `sig`.

        Args:
            theta_lims: (start, end) values of stream angle in radians. Must be
            in -pi to pi. To cross negative x-axis, set non-zero rotation when
            instantiating the stream component.
            mu_r_lims: (start, end) values of stream distance from center.
            sig (float): stream thickness.
            nsmp (int): number of points to sample the angle theta (increase if
                stream looks discretised.

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
        mu_y_smp = mu_r_smp * np.sin(theta_smp)
        nrm_y = stats.norm(mu_y_smp, sig)
        self.p_x_pars = dict(theta_lims=theta_lims,
                             mu_r_lims=mu_r_lims,
                             sig=sig,
                             nsmp=nsmp)
        nt = len(self.cube.get_variable_values('t'))
        tmp = np.array([self.xxp[:,:,np.newaxis] + cube.dx/2.,
                        self.xxp[:,:,np.newaxis] - cube.dx/2.])
        tmp = nrm_x.logcdf(tmp)
        log_p_x = special.logsumexp(tmp.T, -1, b=[1.,-1.]).T
        tmp = np.array([self.yyp[:,:,np.newaxis] + cube.dy/2.,
                        self.yyp[:,:,np.newaxis] - cube.dy/2.])
        tmp = nrm_y.logcdf(tmp)
        log_p_y = special.logsumexp(tmp.T, -1, b=[1.,-1.]).T
        log_p_xy = log_p_x + log_p_y
        log_p_xy = special.logsumexp(log_p_xy, -1)
        log_normalisation = special.logsumexp(log_p_xy)
        log_p_xy -= log_normalisation
        log_p_xy -= np.log(self.cube.dx) + np.log(self.cube.dy)
        log_p_xy = np.broadcast_to(log_p_xy, (nt,)+log_p_xy.shape)
        log_p_xy = np.moveaxis(log_p_xy, 0, -1)
        self.log_p_x_t = log_p_xy

    def set_t_dep(self, t_dep=3.):
        """Set constant depletion timescale

        Args:
            t_dep (float): constant tepletion

        """
        self.t_dep = t_dep * np.ones((self.cube.nx, self.cube.ny))

    def set_mu_v(self, mu_v_lims=[-100,100]):
        """Set mean velocity linearly varying with stream angle

        Args:
            mu_v_lims: (start, end) values of stream velocity

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
        self.mu_v_lims = mu_v_lims
        nt = len(self.cube.get_variable_values('t'))
        mu_v = np.broadcast_to(mu_v, (nt,)+mu_v.shape)
        self.mu_v = mu_v

    def set_sig_v(self, sig_v=100.):
        """Set constant velocity dispersion

        Args:
            sig_v (float): constant std dev of velocity distribution

        """
        nt = self.cube.ssps.par_dims[1]
        size = (nt, self.cube.nx, self.cube.ny)
        self.sig_v = sig_v * np.ones(size)


# end
