import numpy as np
from scipy import stats
from . import parametric

class Stream(parametric.ParametricComponent):
    """A stream component with spatially varying kinematics but spatially uniform enrichment

    The (mass-weighted) joint density of this component can be factorised as
    p(t,x,v,z) = p(t) p(x) p(v|x) p(z|t)
    where the factors are given by:
    - p(t) : a beta distribution (see `set_p_t`),
    - p(x) : a curved line with constant thickness (see `set_p_x`),
    - p(v|x) = Normal(v ; mu_v(x), sig_v(x)) where mean varies linearly along
    stream angle (see `set_mu_v`) and dispersion is constant (see `set_sig_v`)
    - p(z|t) : single chemical evolution track i.. `t_dep` (see `set_p_z_t`).
    - p(z|t) = Normal(z ; mu_z(t, t_dep), sig_z(t, t_dep) i.e. chemical
    enrichment (i.e. metallicity vs t) depends on a single, spatially constant
    depletion timescale t_dep (see `set_t_dep`). The functions mu_z(t,t_dep) and
    sig_z(t,t_dep) are taken from equations 3-10 of Zhu, van de Venn, Leaman et
    al 20.

    Args:
        cube: a pkm.mock_cube.mockCube.
        center (x0,y0): co-ordinates of the component center.
        rotation: angle (radians) between x-axes of component and cube.
        nsmp:

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
        self.p_x_t = pdf[:,:,np.newaxis]

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
        self.mu_v = mu_v[np.newaxis,:,:]

    def set_sig_v(self, sig_v=100.):
        """Set constant velocity dispersion

        Args:
            sig_v (float): constant std dev of velocity distribution

        """
        nt = self.cube.ssps.par_dims[1]
        size = (nt, self.cube.nx, self.cube.ny)
        self.sig_v = sig_v * np.ones(size)


# end
