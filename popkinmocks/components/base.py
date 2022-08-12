import numpy as np
from scipy import stats, special
from abc import ABC, abstractmethod


class baseComponent(ABC):
    """Abstract base class to rerepsent a galaxy component

    A component is specified by it's joint density p(t,x,v,z) over stellar age
    t, 2D position x, line-of-sight velocity v, and metallicity z. Sub-classes
    of `baseComponent` correspond to specific (i) factorisations of the joint
    density and, (ii) implementations of the factors.

    Args:
        cube: a pkm.mock_cube.mockCube.
        p_txvz: array, the 5D density p(t,v,x,z)
        v_edg: array, the edges of velocity bins. Must have size
            `p_txvz.shape[1]+1`

    """
    def __init__(self, cube=None, p_txvz=None, v_edg=None):
        self.cube = cube
        self.p_txvz = p_txvz
        self.v_edg = v_edg

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

    def get_E_v_x(self, light_weighted=False):
        """Get mean velocity map E[p(v|x)]

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        # TODO: implement this for the general case
        pass

    def get_jth_central_moment_v_x(self, j, light_weighted=False):
        """Get j'th central moment of velocity map E[p((v-mu_v)^j|x)]

        Args:
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        # TODO: implement this for the general case
        pass

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

    def evaluate_ybar(self, v_edg=None):
        """Evaluate the noise-free data-cube

        Evaluate the integral
        ybar(x, omega) = int_{-inf}^{inf} s(omega-v ; t,z) P(t,v,x,z) dv dt dz
        where
        omega = ln(wavelength)
        s(omega ; t,z) are stored SSP templates
        This integral is a convolution over velocity v, which we evaluate using
        Fourier transforms (FT). FTs of SSP templates are stored in`ssps.FXw`
        while FTs of the velocity part of the density P(t,v,x,z) are evaluated
        using FFTs here. Sets the result to `self.ybar`.

        Args:
            p_tvxz (array): the density array with dimensions corresponding to
                (t,v,x1,x2,z).
            v_edg (array): array of velocity-bin edges. Number of bins must
                equal p_tvxz.shape[1]. One of the bins must be centered on v=0.

        """
        cube = self.cube
        ssps = cube.ssps
        assert v_edg.size==p_tvxz.shape[1]+1
        assert np.allclose(v_edg[1:]-v_edg[:-1], ssps.dv)
        v = (v_edg[:-1]+v_edg[1:])/2.
        v0_idx = np.where(v==0.)
        assert v0_idx[0].size==1
        v0_idx = v0_idx[0][0]
        assert np.all(np.sort(v)==v)
        p_tvxz = self.get_p_tvxz(v_edg=v_edg, density=True)
        na = np.newaxis
        dtz = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
        F_s_w_tz = np.reshape(ssps.FXw, (-1,)+ssps.par_dims)
        F_s_w_tz = np.moveaxis(F_s_w_tz, -1, 0)
        F_s_w_tz = F_s_w_tz[:,:,na,na,:]
        # move v=0 to correct position for the FFT
        p_tvxz = np.roll(p_tvxz, p_tvxz.shape[1]-v0_idx, axis=1)
        F_p_tvxz = np.fft.rfft(p_tvxz, axis=1) * ssps.dv
        # interpolate FFT to same shape as FFT of SSPs
        interpolator = interpolate.interp1d(
                    np.linspace(0, 1, F_p_tvxz.shape[1]),
                    F_p_tvxz,
                    axis=1,
                    kind='cubic',
                    bounds_error=True)
        F_p_tvxz = interpolator(np.linspace(0, 1, F_s_w_tz.shape[1]))
        F_y = F_s_w_tz*F_p_tvxz
        y = np.fft.irfft(F_y, ssps.n_fft, axis=1)
        y = np.sum(y*dtz, (0,4)) * self.dx * self.dy
        self.ybar = y

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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_t = np.sum(p_tvxz, (1,2,3,4))
        if density:
            p_t /= self.cube.ssps.delta_t
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tv = np.sum(p_tvxz, (2,3,4))
        if density:
            p_tv /= self.construct_volume_element('tv', v_edg=self.v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tx = np.sum(p_tvxz, (1,4))
        if density:
            p_tx /= self.construct_volume_element('tx')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tz = np.sum(p_tvxz, (1,2,3))
        if density:
            p_tz /= self.construct_volume_element('tz')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tvx = np.sum(p_tvxz, 4)
        if density:
            p_tvx /= self.construct_volume_element('tvx', v_edg=self.v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tvz = np.sum(p_tvxz, (2,3))
        if density:
            p_tvz /= self.construct_volume_element('tvz', v_edg=self.v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_txz = np.sum(p_tvxz, 1)
        if density:
            p_txz /= self.construct_volume_element('txz')
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
        p_tvxz = self.p_tvxz
        if density is False:
            p_tvxz *= self.construct_volume_element('tvxz', v_edg=v_edg)
        return p_txz

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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_v = np.sum(p_tvxz, (0,2,3,4))
        if density:
            p_v /= self.construct_volume_element('v', v_edg=v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vx = np.sum(p_tvxz, (0,4))
        if density:
            p_vx /= self.construct_volume_element('vx', v_edg=v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vz = np.sum(p_tvxz, (0,2,3))
        if density:
            p_vz /= self.construct_volume_element('vz', v_edg=v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vxz = np.sum(p_tvxz, 0)
        if density:
            p_vxz /= self.construct_volume_element('vxz', v_edg=v_edg)
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_x = np.sum(p_tvxz, (0,1,4))
        if density:
            p_x /= self.construct_volume_element('x')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_xz = np.sum(p_tvxz, (0,1))
        if density:
            p_xz /= self.construct_volume_element('xz')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_z = np.sum(p_tvxz, (0,1,2,3))
        if density:
            p_z /= self.construct_volume_element('z')
        return p_z
