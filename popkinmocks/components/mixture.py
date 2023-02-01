import numpy as np
from scipy import stats, special
from . import base


class Mixture(base.Component):
    """A mixture component

    The mass-weighted density is a mixture, i.e.

    p(t,x,v,z) = sum_i w_i p_i(t,v,x,z)

    where w_i >= 0 and sum_i w_i = 1.

    Args:
        cube: a pkm.mock_cube.mockCube.
        component_list (list): a list of `pkm.component` objects
        weights (array): weights, must be non-negative and sum-to-one, and same
            length as `component_list`

    """

    def __init__(self, cube=None, component_list=None, weights=None):
        self.cube = cube
        assert len(component_list) == len(weights)
        self.n_cmps = len(component_list)
        self.component_list = component_list
        weights = np.array(weights)
        assert np.all(weights > 0.0)
        assert np.isclose(np.sum(weights), 1.0)
        self.weights = weights

    def evaluate_ybar(self):
        """Evaluate the datacube for this component

        Evaluate the weighted sum

        ybar(x, omega) = sum_i ybar_i

        """
        ybar = np.zeros_like(self.component_list[0].ybar)
        for comp, weight in zip(self.component_list, self.weights):
            ybar += weight * comp.ybar
        self.ybar = ybar

    def get_p(self, which_dist, collapse_cmps=True, density=True, light_weighted=False):
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
                dimensions corresponding to [t,z,x1,x2]. If `collapse_cmps =
                False` the first dimension will index over mixture components.

        """
        log_p = self.get_log_p(
            which_dist,
            collapse_cmps=collapse_cmps,
            density=density,
            light_weighted=light_weighted,
        )
        p = np.exp(log_p)
        return p

    def get_log_p(
        self, which_dist, collapse_cmps=True, density=True, light_weighted=False
    ):
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
                dimensions corresponding to [t,z,x1,x2]. If `collapse_cmps =
                False` the first dimension will index over mixture components.

        """
        is_conditional = "_" in which_dist
        if is_conditional:
            log_p = self._get_log_conditional_distribution(
                which_dist, density=density, light_weighted=light_weighted
            )
        else:
            if light_weighted:
                log_p = self._get_log_marginal_distribution_light_wtd(
                    which_dist, density=density
                )
            else:
                log_p = self._get_log_marginal_distribution_mass_wtd(
                    which_dist, density=density
                )
        if collapse_cmps:
            log_p = special.logsumexp(log_p, 0)
        return log_p

    def _get_log_marginal_distribution_mass_wtd(self, which_dist, density=True):
        """Evaluate component-wise mass-weighted log marginal distributions

        Args:
            which_dist (string): which density to evaluate (this must be
                marginal, not conditional).
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)

        Returns:
            array: first dimension corresponds to components, subsequent
                dimensions align with variables as listed in `which_dist`

        """
        count = 0
        zipped_cmp_wts = zip(self.component_list, self.weights)
        for i, (cmp, w) in enumerate(zipped_cmp_wts):
            log_p_func = getattr(cmp, "_get_log_p_" + which_dist)
            log_pi = np.log(w) + log_p_func(density=density, light_weighted=False)
            if count == 0:
                log_p = np.zeros((self.n_cmps,) + log_pi.shape)
            log_p[i] = log_pi
            count += 1
        return log_p

    def _get_log_marginal_distribution_light_wtd(self, which_dist, density=True):
        """Evaluate component-wise light-weighted log marginal distributions

        Args:
            which_dist (string): which density to evaluate (this must be
                marginal, not conditional).
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)

        Returns:
            array: first dimension corresponds to components, subsequent
                dimensions align with variables as listed in `which_dist`

        """
        na = np.newaxis
        if "v" in which_dist:
            current_dist = "tvxz"
            lw = self.cube.ssps.light_weights[na, :, na, na, na, :]
        else:
            current_dist = "txz"
            lw = self.cube.ssps.light_weights[na, :, na, na, :]
        log_P_mw = self._get_log_marginal_distribution_mass_wtd(
            current_dist, density=False
        )
        log_P_lw = log_P_mw + np.log(lw)
        log_normalisation = special.logsumexp(log_P_lw)
        log_P_lw -= log_normalisation
        # sum over any variables not in the desired distribution
        if "t" not in which_dist:
            log_P_lw = special.logsumexp(log_P_lw, 1)
        # don't swap the order of these next two operations!
        if "x" not in which_dist:
            log_P_lw = special.logsumexp(log_P_lw, (-3, -2))
        if "z" not in which_dist:
            log_P_lw = special.logsumexp(log_P_lw, -1)
        if density:
            volume_element = self.cube.construct_volume_element(which_dist)
            log_P_lw -= np.log(volume_element)
        return log_P_lw

    def _get_log_conditional_distribution(
        self, which_dist, light_weighted=False, density=True
    ):
        """Get conditional distributions

        This is intended to be called only by the `get_p` wrapper method - see
        that docstring for more info.

        Args:
        which_dist (string): which density to evaluate
        density (bool): whether to return probabilty density (True) or the
        volume-element weighted probabilty (False)

        Returns:
        array: the desired distribution.

        """
        assert "_" in which_dist
        dist, marginal = which_dist.split("_")
        # get an alphabetically ordered string for the joint distribution
        joint = "".join(sorted(dist + marginal))
        kwargs = {"density": False, "light_weighted": light_weighted}
        log_p_joint = self.get_log_p(joint, collapse_cmps=False, **kwargs)
        log_p_marginal = self.get_log_p(marginal, collapse_cmps=True, **kwargs)
        # if x is in joint/marginalal, repalace it with xy to account for the
        # fact that x stands for 2D positon (x,y)
        joint = joint.replace("x", "xy")
        marginal = marginal.replace("x", "xy")
        # first dimension in the joint corresponts to component index:
        joint = "i" + joint
        # for each entry in the marginal, find its position in the joint
        old_pos = [joint.find(m0) for m0 in marginal]
        # move the marginal variables to the far right of the joint
        n_marginal = len(marginal)
        new_pos = [-(i + 1) for i in range(n_marginal)][::-1]
        log_p_joint = np.moveaxis(log_p_joint, old_pos, new_pos)
        # get the conditional probability
        log_p_conditional = log_p_joint - log_p_marginal
        if density:
            dvol = self.cube.construct_volume_element(which_dist)
            log_dvol = np.log(dvol)
            log_p_conditional = log_p_conditional - log_dvol
        return log_p_conditional

    def get_mean(self, which_dist, light_weighted=False):
        """Get mean/conditional mean using exact formula for mixtures.

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
        mu_i = np.array(
            [
                cmp.get_mean(which_dist, light_weighted=light_weighted)
                for cmp in self.component_list
            ]
        )
        is_conditional = "_" in which_dist
        if light_weighted:
            rw = self._get_light_reweightings()
        else:
            rw = np.ones(self.n_cmps)
        if is_conditional is False:
            mean = np.sum(self.weights * rw * mu_i.T, -1).T
        else:
            str_split = which_dist.split("_")
            variable_to_mean = str_split[0]
            conditioners = str_split[1]
            w_p_cond = np.array(
                [
                    cmp.get_p(conditioners, light_weighted=light_weighted, density=True)
                    for cmp in self.component_list
                ]
            )
            w_p_cond = (w_p_cond.T * self.weights).T
            if variable_to_mean == "x":
                na = np.newaxis
                w_p_cond = w_p_cond[:, na]
            denom = self.get_p(
                conditioners,
                light_weighted=light_weighted,
                collapse_cmps=True,
                density=True,
            )
            mean = np.sum(w_p_cond.T * rw * mu_i.T, -1).T / denom
        return mean

    def _get_light_reweightings(self):
        """Light of each mixture component, needed for light-weighted moments"""
        p_tz = self.get_p("tz", light_weighted=False, density=False)
        p_tz_i = np.array(
            [
                cmp.get_p("tz", light_weighted=False, density=False)
                for cmp in self.component_list
            ]
        )
        lw = self.cube.ssps.light_weights
        reweightings = np.sum(lw * p_tz_i, (1, 2)) / np.sum(lw * p_tz)
        return reweightings

    def get_central_moment(self, which_dist, j, light_weighted=False):
        """Get central moment or conditional central moment of a 1D distribution

        Uses exact formula for mixture models. See full docstring of
        `self.get_mean` for restrictions on `which_dist` and meaning of output.

        Args:
            which_dist (string): which distribution to take central moment of.
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity

        Returns:
            float/array: the central moment or conditional central moment

        """
        mu = self.get_mean(which_dist, light_weighted=light_weighted)
        mu_i = np.array(
            [
                cmp_i.get_noncentral_moment(
                    which_dist, j, mu, light_weighted=light_weighted
                )
                for cmp_i in self.component_list
            ]
        )
        is_conditional = "_" in which_dist
        if is_conditional is False:
            mu_i = np.moveaxis(mu_i, 0, -1)
            if light_weighted:
                rw = self._get_light_reweightings()
                moment = np.sum(self.weights * rw * mu_i, -1)
            else:
                moment = np.sum(self.weights * mu_i, -1)
        else:
            str_split = which_dist.split("_")
            variable_to_mean = str_split[0]
            conditioners = str_split[1]
            w_p_cond = self.get_p(
                conditioners,
                light_weighted=light_weighted,
                collapse_cmps=False,
                density=True,
            )
            if variable_to_mean == "x":
                na = np.newaxis
                w_p_cond = w_p_cond[:, na]
            moment = np.sum(w_p_cond * mu_i, 0) / np.sum(w_p_cond, 0)
        return moment
