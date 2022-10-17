import numpy as np
from scipy import interpolate
import copy
from functools import partial

class Component(object):
    """A galaxy component

    A component is specified by it's joint density p(t,x,v,z) over stellar age
    t, 2D position x, line-of-sight velocity v, and metallicity z. Sub-classes
    of `Component` correspond to specific (i) factorisations of the joint
    density and, (ii) implementations of the factors.

    Args:
        cube: a pkm.mock_cube.mockCube.
        p_txvz (array): the 5D mass-weighted probabilty density p(t,x1,x2,v,z)

    """
    def __init__(self, cube=None, p_tvxz=None):
        self.cube = cube
        dtvxz = self.construct_volume_element('tvxz')
        if p_tvxz.shape==dtvxz.shape:
            self.p_tvxz = p_tvxz
        else:
            err = f'`p_tvxz` has shape {p_tvxz.shape} '
            err += f'but should have shape {dtvxz.shape}'
            raise ValueError(err)

    def get_p(self,
              which_dist,
              density=True,
              light_weighted=False):
        """Evaluate population-kinematic distributions for this component

        Evaluate marginal or conditional distributions over stellar age t, 2D
        position x, velocity v and metallicity z. The argument `which_dist`
        specifies which distribution to evaluate, where an underscore (if
        present) represents conditioning e.g.
        - `which_dist = 'tv'` --> p(t,v),
        - `which_dist = 'tz_x'` --> p(t,z|x) etc ...
        Variables in `which_dist` must be provided in alphabetical order (on
        either side of the underscore if present).

        Args:
            which_dist (string): which density to evaluate
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

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
                light_weighted=light_weighted)
        else:
            p = self.get_marginal_distribution(
                which_dist,
                density=density,
                light_weighted=light_weighted)
        return p

    def get_marginal_distribution(self,
                                  which_dist,
                                  density=True,
                                  light_weighted=False):
        """Evaluate marginal distributions for this component

        This is intended to be called only by the `get_p` wrapper method - see
        that docstring for more info.

        Args:
            which_dist (string): which density to evaluate.
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array: dimensions align with variables as listed in `which_dist`

        """
        # TODO: check that `which_dist` has no underscore and is alphabetical
        p_func = getattr(self, 'get_p_'+which_dist)
        p = p_func(density=density, light_weighted=light_weighted)
        return p

    def get_conditional_distribution(self,
                                     which_dist,
                                     light_weighted=False,
                                     density=True):
        """Get conditional distributions for this component

        This is intended to be called only by the `get_p` wrapper method - see
        that docstring for more info. Evaluates conditionals as the quotient of
        two marginal distributions.

        Args:
            which_dist (string): which density to evaluate.
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
        array: the desired distribution.

        """
        assert '_' in which_dist
        dist, marginal = which_dist.split('_')
        # if the conditional distribution is hard coded, then use that...
        try:
            p_func = getattr(self, 'get_p_'+which_dist)
            p_cond = p_func(density=density, light_weighted=light_weighted)
        # ... otherwise compute conditional = joint/marginal
        except (AttributeError, NotImplementedError):
            joint = ''.join(sorted(dist+marginal))
            kwargs = {'density':False, 'light_weighted':light_weighted}
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
            p_cond = p_joint/p_marginal
            if density:
                dvol = self.construct_volume_element(which_dist)
                p_cond = p_cond/dvol
        return p_cond

    def construct_volume_element(self, which_dist):
        """Construct volume element for converting densities to probabilties

        Args:
            which_dist (string): which density to evaluate

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
                v_edg = self.cube.v_edg
                da = v_edg[1:] - v_edg[:-1]
            elif var=='x':
                da = np.full(self.cube.nx, self.cube.dx)
            elif var=='y':
                da = np.full(self.cube.ny, self.cube.dy)
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

    def get_mean(self, which_dist, light_weighted=False):
        """Get mean or conditional mean of a 1D distribution

        The `which_dist` string specifies which distribution to mean. If
        `which_dist` contains no underscore, this returns the mean of the
        appropriate marginal distribution e.g.
        - `which_dist = 'v'` returns E(v) = int v p(v) dv
        If `which_dist` contains an underscore, this returns the mean of the
        appropriate conditional distribution e.g.
        - `which_dist = 'v_x'` returns E(v|x) = int v p(v|x) dv
        Only works for distributions of one argument i.e. one variable before
        the underscore in `which_dist`.
        Uses exact calculations if available for a given distribution.

        Args:
            which_dist (string): which distribution to mean. See full docstring.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: if the variable to mean is t, v or z, returns an array
                with shape equal to the shape of the conditioners e.g.
                `which_dist = 'v_tz'` returns array shape (nt, nz)
                and a float if conditional. For x, the returned array has extra
                initial dimension of size 2 for the 2 spatial dimensions.

        """
        variable_to_mean = which_dist.split('_')[0]
        assert len(variable_to_mean)==1
        def get_mean(light_weighted=light_weighted):
            p = self.get_p(which_dist,
                           light_weighted=light_weighted,
                           density=False)
            if variable_to_mean in ['t','v','z']:
                var = self.cube.get_variable_values(variable_to_mean)
                var_pvar_dvar = (var.T * p.T).T # e.g = t p(t|...) dt
                mean = np.sum(var_pvar_dvar, 0)
            else:
                assert variable_to_mean == 'x'
                # right now p = p(x1, x2 | ...) dx1 dx2 but we want
                # p1 = p(x1 | ...) dx1 to get mean(x1) and
                # p2 = p(x2 | ...) dx2 to get mean(x2)
                p1 = np.sum(p, 1)
                p2 = np.sum(p, 0)
                mean = np.array([np.sum(self.cube.x * p1.T, -1).T,
                                 np.sum(self.cube.y * p2.T, -1).T])
            return mean
        # if a mean method is hard coded for a component then use that instead
        if hasattr(self, 'get_mean_'+which_dist):
            get_mean = getattr(self, 'get_mean_'+which_dist)
        mean = get_mean(light_weighted=light_weighted)
        return mean

    def get_central_moment(self, which_dist, j, light_weighted=False):
        """Get central moment or conditional central moment of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output. Uses exact calculations if available for a given
        distribution.

        Args:
            which_dist (string): which distribution to take central moment of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the central moment or conditional central moment

        """
        variable_to_mean = which_dist.split('_')[0]
        assert len(variable_to_mean)==1
        na = np.newaxis
        def get_moment(light_weighted=light_weighted):
            p = self.get_p(which_dist,
                           light_weighted=light_weighted,
                           density=False)
            if variable_to_mean in ['t','v','z']:
                val = self.cube.get_variable_values(variable_to_mean)
                mu = self.get_mean(which_dist, light_weighted=light_weighted)
                tmp = (np.power(val - mu[na].T, j) * p.T).T
                moment = np.sum(tmp, 0)
            else:
                assert variable_to_mean == 'x'
                # right now p = p(x1, x2 | ...) dx1 dx2 but we want
                # p1 = p(x1 | ...) dx1 to get moment(x1) and
                # p2 = p(x2 | ...) dx2 to get moment(x2)
                p1 = np.sum(p, 1)
                p2 = np.sum(p, 0)
                x1 = self.cube.x
                x2 = self.cube.y
                mu1, mu2 = self.get_mean(
                    which_dist,
                    light_weighted=light_weighted)
                tmp1 = np.power(x1 - mu1[na].T, j).T
                tmp2 = np.power(x2 - mu2[na].T, j).T
                moment = np.array([np.sum(tmp1 * p1, 0),
                                   np.sum(tmp2 * p2, 0)])
            return moment
        # if a moment method is hard coded then use that instead
        method_string = f'get_central_moment_{which_dist}'
        if hasattr(self, method_string):
            print(f'using {method_string}')
            get_moment = getattr(self, method_string)
            get_moment = partial(get_moment, j)
        moment = get_moment(light_weighted=light_weighted)
        return moment

    def get_variance(self, which_dist, light_weighted=False):
        """Get variance or conditional variance of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take variance of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the variance or conditional variance

        """
        variance = self.get_central_moment(
            which_dist,
            2,
            light_weighted=light_weighted)
        return variance

    def get_skewness(self, which_dist, light_weighted=False):
        """Get skewness or conditional skewness of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take skewness of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the variance or conditional skewness

        """
        mu_3 = self.get_central_moment(
            which_dist,
            3,
            light_weighted=light_weighted)
        variance = self.get_central_moment(
            which_dist,
            2,
            light_weighted=light_weighted)
        skewness = mu_3/variance**1.5
        return skewness

    def get_kurtosis(self, which_dist, light_weighted=False):
        """Get kurtosis or conditional kurtosis of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take kurtosis of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the kurtosis or conditional kurtosis

        """
        mu_4 = self.get_central_moment(
            which_dist,
            4,
            light_weighted=light_weighted)
        variance = self.get_central_moment(
            which_dist,
            2,
            light_weighted=light_weighted)
        kurtosis = mu_4/variance**2.
        return kurtosis

    def get_excess_kurtosis(self, which_dist, light_weighted=False):
        """Get excess kurtosis or conditional ex-kurtosis of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take ex-kurtosis of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the ex-kurtosis or conditional ex-kurtosis

        """
        kurtosis = self.get_kurtosis(
            which_dist,
            light_weighted=light_weighted)
        return kurtosis - 3.

    def evaluate_ybar(self):
        """Evaluate the datacube for this component

        Evaluate the integral
        ybar(x, omega) = int_{-inf}^{inf} s(omega-v ; t,z) P(t,v,x,z) dv dt dz
        where
        omega = ln(wavelength)
        s(omega ; t,z) are stored SSP templates
        This integral is a convolution over velocity v, which we evaluate using
        FFTs. FFTs of SSP templates are stored in`ssps.FXw` while FFTs of
        (velocity part of) the density P(t,v,x,z) are evaluated here. Sets the
        result to `self.ybar`.

        """
        cube = self.cube
        ssps = cube.ssps
        v_edg = cube.v_edg
        def validate_v_edg():
            error_string = 'Velocity array must be '
            # correct uniform spacing
            check1 = np.allclose(v_edg[1:]-v_edg[:-1], ssps.dv)
            if not check1:
                error_string += '(i) uniformly spaced with dv = SSP resolution'
            # ascending
            v = (v_edg[:-1]+v_edg[1:])/2.
            check2 = np.all(np.sort(v)==v)
            if not check2:
                error_string += '(ii) ascending'
            # zero-centered
            v0_idx = np.where(np.isclose(v, 0.))
            v0_idx = v0_idx[0][0]
            if v.size % 2 == 1:
                check3 = (v0_idx==(v.size-1)/2)
            else:
                check3 = (v0_idx==v.size/2)
            if not check3:
                error_string += '(iii) zero-centered.'
            if check1 and check2 and check3:
                pass
            else:
                raise ValueError(error_string)
        validate_v_edg()
        na = np.newaxis
        dtz = ssps.delta_t[:,na,na,na,na]*ssps.delta_z[na,na,na,na,:]
        F_s_w_tz = np.reshape(ssps.FXw, (-1,)+ssps.par_dims)
        F_s_w_tz = np.moveaxis(F_s_w_tz, -1, 0)
        F_s_w_tz = F_s_w_tz[:,:,na,na,:]
        # move v=0 to correct position for the FFT
        p_tvxz = self.get_p_tvxz(density=True)
        v = (v_edg[:-1]+v_edg[1:])/2.
        v0_idx = np.where(np.isclose(v, 0.))[0][0]
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
        y = np.sum(y*dtz, (0,4)) * self.cube.dx * self.cube.dy
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tv = np.sum(p_tvxz, (2,3,4))
        if density:
            p_tv /= self.construct_volume_element('tv')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tvx = np.sum(p_tvxz, 4)
        if density:
            p_tvx /= self.construct_volume_element('tvx')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_tvz = np.sum(p_tvxz, (2,3))
        if density:
            p_tvz /= self.construct_volume_element('tvz')
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
        p_tvxz = copy.copy(self.p_tvxz)
        if density is False:
            p_tvxz *= self.construct_volume_element('tvxz')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_v = np.sum(p_tvxz, (0,2,3,4))
        if density:
            p_v /= self.construct_volume_element('v')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vx = np.sum(p_tvxz, (0,4))
        if density:
            p_vx /= self.construct_volume_element('vx')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vz = np.sum(p_tvxz, (0,2,3))
        if density:
            p_vz /= self.construct_volume_element('vz')
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
        p_tvxz = self.get_p_tvxz(density=False, light_weighted=light_weighted)
        p_vxz = np.sum(p_tvxz, 0)
        if density:
            p_vxz /= self.construct_volume_element('vxz')
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
