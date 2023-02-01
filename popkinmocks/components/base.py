import numpy as np
from scipy import interpolate, special
import copy
from functools import partial
import os
import dill


class Component(object):
    """A galaxy component specified by the (log) joint density p(t,v,x,z)

    Sub-classes of `Component` correspond to specific choices of p(t,x,v,z).
    Main methods are `get_p` to evaluate probability functions and `get_mean`,
    `get_variance`, `get_skewness`, `get_kurtosis` to get moments.

    Args:
        cube: a pkm.mock_cube.mockCube.
        log_p_txvz (array): 5D array of natural log of mass-weighted probabilty
            density i.e. log p(t,v,x1,x2,z)

    """

    def __init__(self, cube=None, log_p_tvxz=None):
        self.cube = cube
        shape_tvxz = self.cube.get_distribution_shape("tvxz")
        if log_p_tvxz.shape == shape_tvxz:
            self.log_p_tvxz = log_p_tvxz
        else:
            err = f"`log_p_tvxz` has shape {log_p_tvxz.shape} "
            err += f"but should have shape {shape_tvxz}"
            raise ValueError(err)

    def get_p(self, which_dist, density=True, light_weighted=False):
        """Evaluate probability functions for this component

        Evaluate marginal or conditional distributions over stellar age t, 2D
        position x, velocity v and metallicity z. The argument `which_dist`
        specifies which distribution to evaluate, where an underscore (if
        present) represents conditioning e.g.:

        - `which_dist = 'tv'` --> p(t,v),
        - `which_dist = 'tz_x'` --> p(t,z|x) etc

        Variables in `which_dist` must be provided in alphabetical order (on
        either side of the underscore if present).

        Args:
            which_dist (string): valid string for the distribution to evaluate
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
        log_p = self.get_log_p(
            which_dist, density=density, light_weighted=light_weighted
        )
        p = np.exp(log_p)
        return p

    def get_log_p(self, which_dist, density=True, light_weighted=False):
        """Evaluate log probability functions for this component

        Evaluate marginal or conditional distributions over stellar age t, 2D
        position x, velocity v and metallicity z. The argument `which_dist`
        specifies which distribution to evaluate, where an underscore (if
        present) represents conditioning e.g.:

        - `which_dist = 'tv'` --> p(t,v),
        - `which_dist = 'tz_x'` --> p(t,z|x) etc

        Variables in `which_dist` must be provided in alphabetical order (on
        either side of the underscore if present).

        Args:
            which_dist (string): valid string for the distribution to evaluate
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
        is_conditional = "_" in which_dist
        if is_conditional:
            log_p = self._get_log_conditional_distribution(
                which_dist, density=density, light_weighted=light_weighted
            )
        else:
            log_p = self._get_log_marginal_distribution(
                which_dist, density=density, light_weighted=light_weighted
            )
        return log_p

    def _get_log_marginal_distribution(
        self, which_dist, density=True, light_weighted=False
    ):
        """Evaluate marginal distributions for this component

        This is intended to be called only by the `get_log_p` wrapper method -
        see that docstring for more info.

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
        log_p_func = getattr(self, "get_log_p_" + which_dist)
        log_p = log_p_func(density=density, light_weighted=light_weighted)
        return log_p

    def _get_log_conditional_distribution(
        self, which_dist, light_weighted=False, density=True
    ):
        """Get log conditional distributions for this component

        This is intended to be called only by the `get_log_p` wrapper method -
        see that docstring for more info. Evaluates log conditionals as the
        difference of two log marginal distributions.

        Args:
            which_dist (string): which density to evaluate.
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
        array: the desired distribution.

        """
        assert "_" in which_dist
        dist, marginal = which_dist.split("_")
        # if the conditional distribution is hard coded, then use that...
        try:
            log_p_func = getattr(self, "get_log_p_" + which_dist)
            log_p_cond = log_p_func(density=density, light_weighted=light_weighted)
        # ... otherwise compute conditional = joint/marginal
        except (AttributeError, NotImplementedError):
            joint = "".join(sorted(dist + marginal))
            kwargs = {"density": False, "light_weighted": light_weighted}
            log_p_joint = self.get_log_p(joint, **kwargs)
            log_p_marginal = self.get_log_p(marginal, **kwargs)
            # if x is in joint/marginalal, repalace it with xy to account for the
            # fact that x stands for 2D positon (x,y)
            joint = joint.replace("x", "xy")
            marginal = marginal.replace("x", "xy")
            # for each entry in the marginal, find its position in the joint
            old_pos_in_joint = [joint.find(m0) for m0 in marginal]
            # move the marginal variables to the far right of the joint
            n_marginal = len(marginal)
            new_pos_in_joint = [-(i + 1) for i in range(n_marginal)][::-1]
            log_p_joint = np.moveaxis(log_p_joint, old_pos_in_joint, new_pos_in_joint)
            # get the conditional probability (= nan if log_p_marginal = -inf)
            log_p_cond = log_p_joint - log_p_marginal
            if density:
                dvol = self.cube.construct_volume_element(which_dist)
                log_p_cond = log_p_cond - np.log(dvol)
        return log_p_cond

    def get_mean(self, which_dist, light_weighted=False):
        """Get mean or conditional mean of a 1D distribution

        The `which_dist` string specifies which distribution to mean. If
        `which_dist` contains no underscore, this returns the mean of the
        appropriate marginal distribution. If `which_dist` contains an
        underscore, this returns the mean of the conditional distribution e.g.:

        - `which_dist = 'v'` returns E(v) = int v p(v) dv
        - `which_dist = 'v_x'` returns E(v|x) = int v p(v|x) dv

        Only works for distributions of one argument.

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
        variable_to_mean = which_dist.split("_")[0]
        assert len(variable_to_mean) == 1

        def get_mean(light_weighted=light_weighted):
            p = self.get_p(which_dist, light_weighted=light_weighted, density=False)
            if variable_to_mean in ["t", "v", "z"]:
                var = self.cube.get_variable_values(variable_to_mean)
                var_pvar_dvar = (var.T * p.T).T  # e.g = t p(t|...) dt
                mean = np.sum(var_pvar_dvar, 0)
            else:
                assert variable_to_mean == "x"
                # right now p = p(x1, x2 | ...) dx1 dx2 but we want
                # p1 = p(x1 | ...) dx1 to get mean(x1) and
                # p2 = p(x2 | ...) dx2 to get mean(x2)
                p1 = np.sum(p, 1)
                p2 = np.sum(p, 0)
                mean = np.array(
                    [np.sum(self.cube.x * p1.T, -1).T, np.sum(self.cube.y * p2.T, -1).T]
                )
            return mean

        # if a mean method is hard coded for a component then use that instead
        method_string = "get_mean_" + which_dist
        if hasattr(self, method_string):
            get_mean = getattr(self, "get_mean_" + which_dist)
        mean = get_mean(light_weighted=light_weighted)
        return mean

    def get_noncentral_moment(self, which_dist, j, mu, light_weighted=False):
        """Get noncentral moment of 1D distributions

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take central moment of.
            mu (array): center about which to take moment about, with shape
                broadcastable with conditioners of `which_dist` (if present)
            j (int): moment order
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the central moment or conditional central moment

        """
        variable_to_mean = which_dist.split("_")[0]
        assert len(variable_to_mean) == 1
        na = np.newaxis

        def get_moment(light_weighted=light_weighted):
            p = self.get_p(which_dist, light_weighted=light_weighted, density=False)
            if variable_to_mean in ["t", "v", "z"]:
                val = self.cube.get_variable_values(variable_to_mean)
                tmp = (np.power(val - mu[na].T, j) * p.T).T
                moment = np.sum(tmp, 0)
            else:
                assert variable_to_mean == "x"
                # right now p = p(x1, x2 | ...) dx1 dx2 but we want
                # p1 = p(x1 | ...) dx1 to get moment(x1) and
                # p2 = p(x2 | ...) dx2 to get moment(x2)
                p1 = np.sum(p, 1)
                p2 = np.sum(p, 0)
                x1 = self.cube.x
                x2 = self.cube.y
                mu1, mu2 = mu
                tmp1 = np.power(x1 - mu1[na].T, j).T
                tmp2 = np.power(x2 - mu2[na].T, j).T
                moment = np.array([np.sum(tmp1 * p1, 0), np.sum(tmp2 * p2, 0)])
            return moment

        # if a moment method is hard coded then use that instead
        method_string = f"get_noncentral_moment_{which_dist}"
        if hasattr(self, method_string):
            get_moment = getattr(self, method_string)
            get_moment = partial(get_moment, j, mu)
        moment = get_moment(light_weighted=light_weighted)
        return moment

    def get_central_moment(self, which_dist, j, light_weighted=False):
        """Get central moment of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take central moment of.
            j (int) : moment order
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the central moment or conditional central moment

        """
        variable_to_mean = which_dist.split("_")[0]
        assert len(variable_to_mean) == 1
        if variable_to_mean in ["t", "v", "z"]:
            mu = self.get_mean(which_dist, light_weighted=light_weighted)
        else:
            assert variable_to_mean == "x"
            mu1, mu2 = self.get_mean(which_dist, light_weighted=light_weighted)
            mu = np.array([mu1, mu2])
        is_conditional = "_" in which_dist
        if is_conditional:
            conditioners = which_dist.split("_")[1]
            dc = self.cube.construct_volume_element(conditioners)
        moment = self.get_noncentral_moment(
            which_dist, j, mu, light_weighted=light_weighted
        )
        return moment

    def get_variance(self, which_dist, light_weighted=False):
        """Get variance of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take variance of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the variance or conditional variance

        """
        variance = self.get_central_moment(which_dist, 2, light_weighted=light_weighted)
        return variance

    def get_skewness(self, which_dist, light_weighted=False):
        """Get skewness of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take skewness of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the variance or conditional skewness

        """
        mu_3 = self.get_central_moment(which_dist, 3, light_weighted=light_weighted)
        variance = self.get_central_moment(which_dist, 2, light_weighted=light_weighted)
        skewness = mu_3 / variance**1.5
        return skewness

    def get_kurtosis(self, which_dist, light_weighted=False):
        """Get kurtosis of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take kurtosis of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the kurtosis or conditional kurtosis

        """
        mu_4 = self.get_central_moment(which_dist, 4, light_weighted=light_weighted)
        variance = self.get_central_moment(which_dist, 2, light_weighted=light_weighted)
        kurtosis = mu_4 / variance**2.0
        return kurtosis

    def get_excess_kurtosis(self, which_dist, light_weighted=False):
        """Get excess kurtosis of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take ex-kurtosis of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the ex-kurtosis or conditional ex-kurtosis

        """
        kurtosis = self.get_kurtosis(which_dist, light_weighted=light_weighted)
        return kurtosis - 3.0

    def evaluate_ybar(self):
        """Evaluate the datacube for this component

        This evaluates the full integral assuming a 5D joint density i.e.:

        $$
        \\bar{y}(x, \omega) = int S(\omega-v ; t,z) p(t,v,x,z) dv dt dz
        $$

        where $\omega = \log \lambda$. This integral is a convolution over
        velocity v, which we evaluate using FFTs. FFTs of SSP templates are
        stored in`ssps.FXw` while FFTs of (the velocity part of) the density
        p(t,v,x,z) are evaluated here. Sets the result to `self.ybar`.

        """
        cube = self.cube
        ssps = cube.ssps
        v_edg = cube.v_edg

        def validate_v_edg():
            error_string = "Velocity array must be "
            # correct uniform spacing
            check1 = np.allclose(v_edg[1:] - v_edg[:-1], ssps.dv)
            if not check1:
                error_string += "(i) uniformly spaced with dv = SSP resolution"
            # ascending
            v = (v_edg[:-1] + v_edg[1:]) / 2.0
            check2 = np.all(np.sort(v) == v)
            if not check2:
                error_string += "(ii) ascending"
            # zero-centered
            v0_idx = np.where(np.isclose(v, 0.0))
            v0_idx = v0_idx[0][0]
            if v.size % 2 == 1:
                check3 = v0_idx == (v.size - 1) / 2
            else:
                check3 = v0_idx == v.size / 2
            if not check3:
                error_string += "(iii) zero-centered."
            if check1 and check2 and check3:
                pass
            else:
                raise ValueError(error_string)

        validate_v_edg()
        na = np.newaxis
        dtz = ssps.delta_t[:, na, na, na, na] * ssps.delta_z[na, na, na, na, :]
        F_s_w_tz = np.reshape(ssps.FXw, (-1,) + ssps.par_dims)
        F_s_w_tz = np.moveaxis(F_s_w_tz, -1, 0)
        F_s_w_tz = F_s_w_tz[:, :, na, na, :]
        # move v=0 to correct position for the FFT
        p_tvxz = self.get_p("tvxz", density=True, light_weighted=False)
        v = (v_edg[:-1] + v_edg[1:]) / 2.0
        v0_idx = np.where(np.isclose(v, 0.0))[0][0]
        p_tvxz = np.roll(p_tvxz, p_tvxz.shape[1] - v0_idx, axis=1)
        F_p_tvxz = np.fft.rfft(p_tvxz, axis=1) * ssps.dv
        # interpolate FFT to same shape as FFT of SSPs
        interpolator = interpolate.interp1d(
            np.linspace(0, 1, F_p_tvxz.shape[1]),
            F_p_tvxz,
            axis=1,
            kind="cubic",
            bounds_error=True,
        )
        F_p_tvxz = interpolator(np.linspace(0, 1, F_s_w_tz.shape[1]))
        F_y = F_s_w_tz * F_p_tvxz
        y = np.fft.irfft(F_y, ssps.n_fft, axis=1)
        y = np.sum(y * dtz, (0, 4)) * self.cube.dx * self.cube.dy
        self.ybar = y

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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_t = special.logsumexp(log_p_tvxz, (1, 2, 3, 4))
        if density:
            log_p_t -= np.log(self.cube.ssps.delta_t)
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tv = special.logsumexp(log_p_tvxz, (2, 3, 4))
        if density:
            log_p_tv -= np.log(self.cube.construct_volume_element("tv"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tx = special.logsumexp(log_p_tvxz, (1, 4))
        if density:
            log_p_tx -= np.log(self.cube.construct_volume_element("tx"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tz = special.logsumexp(log_p_tvxz, (1, 2, 3))
        if density:
            log_p_tz -= np.log(self.cube.construct_volume_element("tz"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tvx = special.logsumexp(log_p_tvxz, 4)
        if density:
            log_p_tvx -= np.log(self.cube.construct_volume_element("tvx"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tvz = special.logsumexp(log_p_tvxz, (2, 3))
        if density:
            log_p_tvz -= np.log(self.cube.construct_volume_element("tvz"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_txz = special.logsumexp(log_p_tvxz, 1)
        if density:
            log_p_txz -= np.log(self.cube.construct_volume_element("txz"))
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
        log_p_tvxz = copy.copy(self.log_p_tvxz)
        log_dtvxz = np.log(self.cube.construct_volume_element("tvxz"))
        log_p_tvxz += log_dtvxz
        if light_weighted:
            na = np.newaxis
            log_lw = np.log(self.cube.ssps.light_weights[:, na, na, na, :])
            log_p_tvxz += log_lw
            log_normalisation = special.logsumexp(log_p_tvxz)
            log_p_tvxz -= log_normalisation
        if density:
            log_p_tvxz -= log_dtvxz
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_v = special.logsumexp(log_p_tvxz, (0, 2, 3, 4))
        if density:
            log_p_v -= np.log(self.cube.construct_volume_element("v"))
        return log_p_v

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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vx = special.logsumexp(log_p_tvxz, (0, 4))
        if density:
            log_p_vx -= np.log(self.cube.construct_volume_element("vx"))
        return log_p_vx

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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vz = special.logsumexp(log_p_tvxz, (0, 2, 3))
        if density:
            log_p_vz -= np.log(self.cube.construct_volume_element("vz"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vxz = special.logsumexp(log_p_tvxz, 0)
        if density:
            log_p_vxz -= np.log(self.cube.construct_volume_element("vxz"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_x = special.logsumexp(log_p_tvxz, (0, 1, 4))
        if density:
            log_p_x -= np.log(self.cube.construct_volume_element("x"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_xz = special.logsumexp(log_p_tvxz, (0, 1))
        if density:
            log_p_xz -= np.log(self.cube.construct_volume_element("xz"))
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
        log_p_tvxz = self.get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_z = special.logsumexp(log_p_tvxz, (0, 1, 2, 3))
        if density:
            log_p_z -= np.log(self.cube.construct_volume_element("z"))
        return log_p_z

    def dill_dump(self, fname, direc=None):
        """Save the component using dill

        Makes the directory if it does not already exist. Saves component there
        using dill.

        Args:
            fname (string): filename
            direc (string, optional): directory name

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        with open(direc + fname, "wb") as f:
            dill.dump(self, f)

    def save_numpy(self, fname, yobs=None, direc=None):
        """Save the component in `npz` format. Deprecated for `dill_dump`.

        Saves various quantities about the cube and component in a `npz` file.
        The density $p(t,v,x,z)$ is rearranged to $p(x,v,z,t)$ for backward
        compatibility with v0.0.

        Args:
            fname (string): filename
            direc (string, optional): directory name
            yobs (array, optional): the observed datacube, i.e. including noise

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        v_edg = self.cube.v_edg
        u_edg = np.log(1.0 + v_edg / self.ssps.speed_of_light)
        p_tvxz = self.get_p("tvxz", collapse_cmps=True, density=True)
        f_xvtz = np.moveaxis(p_tvxz, [0, 1, 2, 3, 4], [4, 2, 0, 1, 3])
        np.savez(
            direc + fname,
            yobs=yobs,
            nx1=self.cube.nx,
            nx2=self.cube.ny,
            x1rng=self.cube.x1rng,
            x2rng=self.cube.x2rng,
            S=self.cube.ssps.Xw.reshape((-1,) + self.cube.ssps.par_dims),
            w=self.cube.ssps.w,
            z_bin_edges=self.cube.ssps.par_edges[0],
            t_bin_edges=self.cube.ssps.par_edges[1],
            ybar=self.ybar,
            v_edg=v_edg,
            f_xvtz=f_xvtz,
        )
