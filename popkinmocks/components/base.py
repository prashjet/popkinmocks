import numpy as np
from scipy import interpolate, special, stats
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
        log_p_func = getattr(self, "_get_log_p_" + which_dist)
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
            log_p_func = getattr(self, "_get_log_p_" + which_dist)
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

    def get_dispersion(self, which_dist, light_weighted=False):
        """Get standard deviation of a 1D distribution

        See full docstring of `self.get_mean` for restrictions on `which_dist`
        and meaning of output.

        Args:
            which_dist (string): which distribution to take SD of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the sd or conditional sd

        """
        variance = self.get_central_moment(which_dist, 2, light_weighted=light_weighted)
        sd = variance**0.5
        return sd

    def get_normalised_central_moment(self, which_dist, j, light_weighted=False):
        """Get normalised central moment of a 1D distribution

        (Normalised central moment)(j) = (central moment)(j)/dispersion**j 

        Args:
            which_dist (string): which distribution to take normalised central 
                moment of.
            j (int) : moment order
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the normalised central (/conditional?) moment

        """
        central_moment = self.get_central_moment(
            which_dist, j, light_weighted=light_weighted
            )
        dispersion = self.get_dispersion(
            which_dist, light_weighted=light_weighted
            )
        normalised_central_moment = central_moment/dispersion**j
        return normalised_central_moment
    
    def get_normalised_central_moment_normal(self, j):
        if j%2 == 0:
            nrm_mom = special.factorial2(j-1)
        else:
            nrm_mom = 0.
        return nrm_mom

    def get_excess_central_moment(self, which_dist, j, light_weighted=False):
        nrm_central_moment = self.get_normalised_central_moment(
            which_dist, j, light_weighted=light_weighted
            )
        nrm_moment_normal = self.get_normalised_central_moment_normal(j)
        excess_central_moment = nrm_central_moment - nrm_moment_normal
        return excess_central_moment

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
        skewness = self.get_normalised_central_moment(
            which_dist, 3, light_weighted=light_weighted
            )
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
        kurtosis = self.get_normalised_central_moment(
            which_dist, 4, light_weighted=light_weighted
            )
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
        exkurtosis = self.get_excess_central_moment(
            which_dist, 4, light_weighted=light_weighted
            )
        return exkurtosis

    def get_covariance(self, which_dist, light_weighted=False):
        """Get covariance or conditional covariance of a 2D distribution

        For two variables a,b out of ('t', 'v', 'x', 'z') get their covariance

        .. math::
            \mathbb{Cov}(a,b)=\int p(a,b)(a-\mathbb{E}(a))(b-\mathbb{E}(b))\;da\;db

        or conditional covariance given a third variable c,

        .. math::
            \mathbb{Cov}(a,b|c)=\int p(a,b|c)(a-\mathbb{E}(a|c))(b-\mathbb{E}(b|c))\;da\;db

        where integrals estimated via summation over discretisation bins in a/b.

        Args:
            which_dist (string): a bivariate distribution to take covariance of
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the covariance or conditional covariance

        """

        if "_" in which_dist:
            is_conditional = True
            dependent_vars, conditioners = which_dist.split("_")
        else:
            is_conditional = False
            dependent_vars = which_dist
        if len(dependent_vars) != 2:
            raise ValueError("Should be exactly two dependent variables")
        means = []
        values = []
        for i in [0, 1]:
            dvar = dependent_vars[i]
            tmp_dist = dvar
            if is_conditional:
                tmp_dist += f"_{conditioners}"
            means += [self.get_mean(tmp_dist, light_weighted=light_weighted)]
            if dvar != "x":
                values += [self.cube.get_variable_values(dvar)]
            else:
                x1 = self.cube.get_variable_values("x1")
                x2 = self.cube.get_variable_values("x2")
                values += [(x1, x2)]
        p = self.get_p(which_dist, density=False, light_weighted=light_weighted)
        na = np.newaxis
        if "x" not in dependent_vars:
            delta_var0 = (values[0] - means[0][na].T).T
            delta_var1 = (values[1] - means[1][na].T).T
            integrand = p * delta_var0[:, na] * delta_var1[na, :]
            covariance = np.sum(integrand, (0, 1))
        else:
            idx_x = dependent_vars.find("x")
            idx_notx = 1 - idx_x  # i.e 0 if idx_x=1, 1 otherwise
            delta_x1 = (values[idx_x][0] - means[idx_x][0][na].T).T
            delta_x2 = (values[idx_x][1] - means[idx_x][1][na].T).T
            delta_notx = (values[idx_notx] - means[idx_notx][na].T).T
            if idx_x == 0:
                intgrnd_x1 = np.sum(p, 1) * delta_x1[:, na] * delta_notx[na, :]
                intgrnd_x2 = np.sum(p, 0) * delta_x2[:, na] * delta_notx[na, :]
            else:
                intgrnd_x1 = np.sum(p, 2) * delta_x1[na, :] * delta_notx[:, na]
                intgrnd_x2 = np.sum(p, 1) * delta_x2[na, :] * delta_notx[:, na]
            covar_x1 = np.sum(intgrnd_x1, (0, 1))
            covar_x2 = np.sum(intgrnd_x2, (0, 1))
            covariance = np.array([covar_x1, covar_x2])
        return covariance

    def get_correlation(self, which_dist, light_weighted=False):
        """Get correlation or conditional correlation of a 2D distribution

        .. math::
            \mathbb{Cor}(a,b|c) = \\frac{\mathbb{Cov}(a,b|c)}{\sqrt{\mathbb{Var}(a|c)\mathbb{Var}(b|c)}}

        Args:
            which_dist (string): a bivariate distribution to take covariance of
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the correlation or conditional correlation

        """
        if "_" in which_dist:
            is_conditional = True
            dependent_vars, conditioners = which_dist.split("_")
        else:
            is_conditional = False
            dependent_vars = which_dist
        if len(dependent_vars) != 2:
            raise ValueError("Should be exactly two dependent variables")
        covariance = self.get_covariance(which_dist, light_weighted=light_weighted)
        tmp0 = dependent_vars[0]
        if is_conditional:
            tmp0 += f"_{conditioners}"
        variance0 = self.get_variance(tmp0, light_weighted=light_weighted)
        tmp1 = dependent_vars[1]
        if is_conditional:
            tmp1 += f"_{conditioners}"
        variance1 = self.get_variance(tmp1, light_weighted=light_weighted)
        correlation = covariance / (variance0 * variance1) ** 0.5
        return correlation

    def get_l_moment(self, which_dist, j, light_weighted=False):
        """Get j'th L-moment of a 1D distributions

        L-moments are robust alternatives to conventional moments based on
        order statistics. Implementation is based on Eq. 2.4 of  `Hoskins 90`_

        .. _Hoskins 90: https://belinra.inrae.fr/doc_num.php?explnum_id=4675

        Args:
            which_dist (string): distribution to take L-moment of
            j (int): moment order
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the central moment or conditional central moment

        """
        var = which_dist.split("_")[0]
        assert len(var) == 1
        if len(which_dist) == 1:
            marginal = True
        else:
            marginal = False
            conditioners = which_dist.split("_")[1]
        # get CDF of distribution
        F = self.get_p(which_dist, density=False, light_weighted=light_weighted)
        F = np.cumsum(F, axis=0)
        # append an initial 0 to CDF
        if marginal:
            initial_zero = np.array([0.])
        else:
            cond_shape = self.cube.get_distribution_shape(conditioners)
            initial_zero = np.zeros((1,)+cond_shape)
        F = np.concatenate((initial_zero, F))
        dF = F[1:] - F[:-1]
        # get L-moment
        poly = special.legendre(j-1)
        g = poly(2*F-1.)
        x = self.cube.get_variable_edges(var)
        y = (x * g.T).T
        l_moment = np.sum((y[:-1] + y[1:])/2.*dF, axis=0)
        return l_moment

    def get_l_mean(self, which_dist, light_weighted=False):
        """Get L-mean of a 1D distribution

        The L-mean is a robust alternative to the mean. See docstring of 
        `self.get_l_moment` for more. The docstring of `self.get_mean` lists
        restrictions on `which_dist` and describes the shape of the output.

        Args:
            which_dist (string): which distribution to take L-mean of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the L-mean or conditional L-mean

        """
        l_mean = self.get_l_moment(
            which_dist, j=1, light_weighted=light_weighted
            )
        return l_mean

    def get_l_dispersion(self, which_dist, light_weighted=False):
        """Get L-dispersion of a 1D distribution

        The L-dispersion is a robust alternative to dispersion. See docstr of 
        `self.get_l_moment` for more. The docstring of `self.get_mean` lists
        restrictions on `which_dist` and describes the shape of the output.

        Args:
            which_dist (string): which distribution to take L-dispersion of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the L-dispersion or conditional L-dispersion

        """
        l_dispersion = self.get_l_moment(
            which_dist, j=2, light_weighted=light_weighted
            )
        return l_dispersion

    def get_normalised_l_moment(self, which_dist, j, light_weighted=False):
        """Get normalised L-moment of a 1D distribution

        L-moments are robust alternatives to moments (see docstring of 
        `self.get_l_moment` for more). Normalised L-moments are dimensionless
        quantities, scaled by the L-dispersion. This is analogous to e.g. 
        skewness being dimensionless third central momnent, scaled by standard
        deviation. The docstring of `self.get_mean` lists restrictions on 
        `which_dist` and describes the shape of the output.

        Args:
            which_dist (string): which distribution to take L-dispersion of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the normalised L-moment or conditional L-moment

        """
        l_dispersion = self.get_l_dispersion(
            which_dist, light_weighted=light_weighted
            )
        lambda_j = self.get_l_moment(
            which_dist, j=j, light_weighted=light_weighted
            )
        return lambda_j/l_dispersion

    def get_l_moment_normal(self, j):
        """Get L-moment of a unit normal distribution

        Args:
            j (int): order of moment

        Returns:
            float: the L-moment of a unit normal distribution

        """
        nrm = stats.norm(0,1)
        poly = special.legendre(j-1)
        F_edg = np.linspace(0, 1, 100000)
        dF = F_edg[1:] - F_edg[:-1]
        F_cnt = (F_edg[:-1] + F_edg[1:])/2.
        x = nrm.ppf(F_cnt)
        g = poly(2*F_cnt-1.)
        y = x*g
        l_moment_normal = np.sum(y*dF)
        if np.isclose(l_moment_normal, 0.):
            l_moment_normal = 0.
        return l_moment_normal

    def get_normalised_l_moment_normal(self, j):
        """Get normalised L-moment of a unit normal distribution

        Args:
            j (int): order of moment

        Returns:
            float: the normalised L-moment of a unit normal distribution

        """
        l_moment_normal = self.get_l_moment_normal(j)
        l2_normal = self.get_l_moment_normal(2)
        return l_moment_normal/l2_normal

    def get_excess_l_moment(self, which_dist, j, light_weighted=False):
        """Get excess L-moment of a 1D distrbution
        
        (Even) L-moments are not zero-centered for normal distributions. We
        define the excess L-moment to be the normalised L-moment subtract the
        equivalent order normalised L-moment of a normal distributions (in
        analgy with excess kurtosis vs kurtosis).
        
        Args:
            which_dist (string): which distribution to take ex. L-moment of
            j (int): order of moment
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the exces L-moment of a unit normal distribution

        """
        l_moment = self.get_normalised_l_moment(
            which_dist,
            j,
            light_weighted=light_weighted)
        l_moment_normal = self.get_normalised_l_moment_normal(j)
        return l_moment - l_moment_normal
        
    def get_l_skewness(self, which_dist, light_weighted=False):
        """Get L-skewness of a 1D distrbution
        
        Args:
            which_dist (string): which distribution to take L-skewness of
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the L-skewness of the distribution

        """
        l_skewness = self.get_normalised_l_moment(
            which_dist,
            3,
            light_weighted=light_weighted)
        return l_skewness

    def get_l_kurtosis(self, which_dist, light_weighted=False):
        """Get L-kurtosis of a 1D distrbution
        
        Args:
            which_dist (string): which distribution to take L-kurtosis of
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the L-kurtosis of the distribution

        """
        l_kurtosis = self.get_normalised_l_moment(
            which_dist,
            4,
            light_weighted=light_weighted)
        return l_kurtosis

    def get_excess_l_kurtosis(self, which_dist, light_weighted=False):
        """Get excess L-kurtosis of a 1D distrbution
        
        Args:
            which_dist (string): which distribution to get excess L-kurtosis of
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the excess L-kurtosis of the distribution

        """
        excess_l_kurtosis = self.get_excess_l_moment(
            which_dist,
            4,
            light_weighted=light_weighted)
        return excess_l_kurtosis

    def evaluate_ybar(self, batch="none"):
        """Evaluate the datacube for this component

        This evaluates the full integral assuming a 5D joint density i.e.:

        .. math::
            \\bar{y}(x, \omega) = \int S(\omega-v ; t,z) p(t,v,x,z) dv dt dz

        where omega is log lambda. This integral is a convolution over
        velocity v, which we evaluate using FFTs. FFTs of SSP templates are
        stored in`ssps.FXw` while FFTs of (the velocity part of) the density
        p(t,v,x,z) are evaluated here. Sets the result to `self.ybar`.

        Args:
            batch (bool, optional): evaluate datacube row by row, default False

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
        F_s_w_tz = np.reshape(ssps.FXw, (-1,) + ssps.par_dims)
        F_s_w_tz = np.moveaxis(F_s_w_tz, -1, 0)
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
        if batch == "none":
            dtz = ssps.delta_t[:, na, na, na, na] * ssps.delta_z[na, na, na, na, :]
            F_s_w_tz = F_s_w_tz[:, :, na, na, :]
            F_y = F_s_w_tz * F_p_tvxz
            y = np.fft.irfft(F_y, ssps.n_fft, axis=1)
            y = np.sum(y * dtz, (0, 4)) * self.cube.dx * self.cube.dy
        elif batch == "column":
            nl = ssps.Xw.shape[0]
            y = np.zeros((nl, self.cube.nx, self.cube.ny), dtype=np.float64)
            dtz = ssps.delta_t[:, na, na, na] * ssps.delta_z[na, na, na, :]
            F_s_w_tz = F_s_w_tz[:, :, na, :]
            for i in range(self.cube.ny):
                F_y_i = F_s_w_tz * F_p_tvxz[:, :, :, i, :]
                y_i = np.fft.irfft(F_y_i, ssps.n_fft, axis=1)
                y[:, :, i] = np.sum(y_i * dtz, (0, 3)) * self.cube.dx * self.cube.dy
        elif batch == "spaxel":
            nl = ssps.Xw.shape[0]
            y = np.zeros((nl, self.cube.nx, self.cube.ny), dtype=np.float64)
            dtz = ssps.delta_t[:, na, na] * ssps.delta_z[na, na, :]
            for i in range(self.cube.nx):
                for j in range(self.cube.ny):
                    F_y_ij = F_s_w_tz * F_p_tvxz[:, :, i, j, :]
                    y_ij = np.fft.irfft(F_y_ij, ssps.n_fft, axis=1)
                    y[:, i, j] = (
                        np.sum(y_ij * dtz, (0, 2)) * self.cube.dx * self.cube.dy
                    )
        else:
            raise ValueError("Unknown option for batch")
        self.ybar = y

    def _get_log_p_t(self, density=True, light_weighted=False):
        """Get log p(t)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_t = special.logsumexp(log_p_tvxz, (1, 2, 3, 4))
        if density:
            log_p_t -= np.log(self.cube.ssps.delta_t)
        return log_p_t

    def _get_log_p_tv(self, density=True, light_weighted=False):
        """Get log p(t,v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tv = special.logsumexp(log_p_tvxz, (2, 3, 4))
        if density:
            log_p_tv -= np.log(self.cube.construct_volume_element("tv"))
        return log_p_tv

    def _get_log_p_tx(self, density=True, light_weighted=False):
        """Get log p(t,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tx = special.logsumexp(log_p_tvxz, (1, 4))
        if density:
            log_p_tx -= np.log(self.cube.construct_volume_element("tx"))
        return log_p_tx

    def _get_log_p_tz(self, density=True, light_weighted=False):
        """Get log p(t,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tz = special.logsumexp(log_p_tvxz, (1, 2, 3))
        if density:
            log_p_tz -= np.log(self.cube.construct_volume_element("tz"))
        return log_p_tz

    def _get_log_p_tvx(self, density=True, light_weighted=False):
        """Get log p(t,v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tvx = special.logsumexp(log_p_tvxz, 4)
        if density:
            log_p_tvx -= np.log(self.cube.construct_volume_element("tvx"))
        return log_p_tvx

    def _get_log_p_tvz(self, density=True, light_weighted=False):
        """Get log p(t,v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_tvz = special.logsumexp(log_p_tvxz, (2, 3))
        if density:
            log_p_tvz -= np.log(self.cube.construct_volume_element("tvz"))
        return log_p_tvz

    def _get_log_p_txz(self, density=True, light_weighted=False):
        """Get log p(t,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_txz = special.logsumexp(log_p_tvxz, 1)
        if density:
            log_p_txz -= np.log(self.cube.construct_volume_element("txz"))
        return log_p_txz

    def _get_log_p_tvxz(self, density=True, light_weighted=False):
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

    def _get_log_p_v(self, density=True, light_weighted=False):
        """Get log p(v)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_v = special.logsumexp(log_p_tvxz, (0, 2, 3, 4))
        if density:
            log_p_v -= np.log(self.cube.construct_volume_element("v"))
        return log_p_v

    def _get_log_p_vx(self, density=True, light_weighted=False):
        """Get log p(v,x)

        Args:
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vx = special.logsumexp(log_p_tvxz, (0, 4))
        if density:
            log_p_vx -= np.log(self.cube.construct_volume_element("vx"))
        return log_p_vx

    def _get_log_p_vz(self, density=True, light_weighted=False):
        """Get log p(v,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vz = special.logsumexp(log_p_tvxz, (0, 2, 3))
        if density:
            log_p_vz -= np.log(self.cube.construct_volume_element("vz"))
        return log_p_vz

    def _get_log_p_vxz(self, density=True, light_weighted=False):
        """Get log p(v,x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_vxz = special.logsumexp(log_p_tvxz, 0)
        if density:
            log_p_vxz -= np.log(self.cube.construct_volume_element("vxz"))
        return log_p_vxz

    def _get_log_p_x(self, density=True, light_weighted=False):
        """Get log p(x)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_x = special.logsumexp(log_p_tvxz, (0, 1, 4))
        if density:
            log_p_x -= np.log(self.cube.construct_volume_element("x"))
        return log_p_x

    def _get_log_p_xz(self, density=True, light_weighted=False):
        """Get log p(x,z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
        log_p_xz = special.logsumexp(log_p_tvxz, (0, 1))
        if density:
            log_p_xz -= np.log(self.cube.construct_volume_element("xz"))
        return log_p_xz

    def _get_log_p_z(self, density=True, light_weighted=False):
        """Get log p(z)

        Args:
            density (bool): whether to return probabilty density (True) or the
            volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
            mass-weighted (False) quantity

        Returns:
            array

        """
        log_p_tvxz = self._get_log_p_tvxz(density=False, light_weighted=light_weighted)
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
        The density p(t,v,x,z) is rearranged to p(x,v,z,t) for backward
        compatibility with v0.0.

        Args:
            fname (string): filename
            direc (string, optional): directory name
            yobs (array, optional): the observed datacube, i.e. including noise

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        v_edg = self.cube.v_edg
        u_edg = np.log(1.0 + v_edg / self.cube.ssps.speed_of_light)
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
