import numpy as np
from scipy import stats, interpolate, optimize, special
import matplotlib.pyplot as plt
import os
import dill

class IFUCube(object):
    """An IFU representing a galaxy composed of a mixture of components

    Add components to the galaxy datacube using the `combine_components` method.

    Args:
        ssps : SPP template, in a `pkm.model_grids.milesSSPs` object
        nx (int): number of pixels in x-dimension
        ny (int): number of pixels in y-dimension
        xrng (tuple): start/end co-ordinates in x-direction
        yrng (tuple): start/end co-ordinates in x-direction

    """

    def __init__(self,
                 ssps=None,
                 nx=300,
                 ny=299,
                 xrng=(-1,1),
                 yrng=(-1,1)):
        self.ssps = ssps
        self.nx = nx
        self.ny = ny
        self.xrng = xrng
        self.yrng = yrng
        self.x = np.linspace(*xrng, nx)
        # flip y array to be compatible with plt.imshow
        self.y = np.linspace(*yrng, ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        xx, yy = np.meshgrid(self.x, self.y, indexing='ij')
        self.xx = xx
        self.yy = yy

    def combine_components(self,
                           component_list=None,
                           weights=None):
        """Combine galaxy components as a mixture model

        Stores the components, weights and evaluates to mixture ybar.

        Args:
            component_list (list): list of 'pkm.components.component` objects
            weights (array-like): simplex-weights

        """
        assert len(component_list)==len(weights)
        self.n_cmps = len(component_list)
        self.component_list = component_list
        weights = np.array(weights)
        assert np.all(weights>0.)
        assert np.isclose(np.sum(weights), 1.)
        self.weights = weights
        ybar = 0. * component_list[0].ybar
        for comp, weight in zip(component_list, weights):
            ybar += weight * comp.ybar
        self.ybar = ybar

    def add_noise(self, snr=100.):
        """Add noise with SNR which is consant with position and wavelength

        yobs ~ Norm(ybar, ybar/snr)

        Args:
            snr : the desired signal-to-noise ratio

        """
        nrm = stats.norm(0.*self.ybar, self.ybar/snr)
        self.snr = snr
        self.noise = nrm.rvs()
        self.yobs = self.ybar + self.noise

    def add_sqrt_signal_noise(self, noise_constant=1.):
        """Add noise which scales by sqrt of the signal

        yobs ~ Norm(ybar, noise_constant*sqrt(ybar))

        Args:
            noise_constant : scaing constant for the noise

        """
        nrm = stats.norm(0, self.noise_constant*self.ybar**0.5)
        self.noise_constant = noise_constant
        self.noise = nrm.rvs()
        self.yobs = self.ybar + self.noise

    def save_data(self,
                  direc=None,
                  fname=None):
        """Save the IFUCube object

        Args:
            direc (string): directory name
            fname (string): filename

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        with open(direc + fname, 'wb') as f:
            dill.dump(self, f)

    def save_numpy(self,
                   v_edg=None,
                   direc=None,
                   fname=None):
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        if v_edg is None:
            v_edg = np.arange(-1000, 1001, self.ssps.dv)
        u_edg = np.log(1. + v_edg/self.ssps.speed_of_light)
        p_tvxz = self.get_p('tvxz',
                            collapse_cmps=True,
                            density=True,
                            v_edg=v_edg)
        f_xvtz = np.moveaxis(p_tvxz, [0,1,2,3,4], [4,2,0,1,3])
        np.savez(direc + fname,
                 nx1=self.nx,
                 nx2=self.ny,
                 x1rng=self.xrng,
                 x2rng=self.yrng,
                 S=self.ssps.Xw.reshape((-1,)+self.ssps.par_dims),
                 w=self.ssps.w,
                 z_bin_edges=self.ssps.par_edges[0],
                 t_bin_edges=self.ssps.par_edges[1],
                 ybar=self.ybar,
                 y=self.yobs,
                 v_edg=v_edg,
                 f_xvtz=f_xvtz)

    def get_p(self,
              which_dist,
              collapse_cmps=False,
              density=True,
              light_weighted=False,
              v_edg=None):
        """Evaluate densities of the galaxy

        5D densities over: stellar age (t), 2D position (x), velocity (v), and
        metallicity (z). Which density to evaluate is specificed by `which_dist`
        which can be one of:
        - 't'
        - 'x_t'
        - 'tx'
        - 'z_tx'
        - 'txz'
        - 'x'
        - 'z'
        - 'tz_x'
        - 'tz'
        - 'tvxz'
        - 'vx'
        - 'v_x'
        - 'v'
        where the underscore (if-present) represents conditioning i.e. calling
        `get_p('tz_x')` will return p(t,z|x). This calls the `get_p...` methods
        of the galaxy's consistituent components (n.b. these must be defined!)

        Args:
            which_dist (string): which density to evaluate
            collapse_cmps (bool): whether to collapse component densities
                together (True) or leave them in-tact (False)
            density (bool): whether to return probabilty density (True) or the
                volume-element weighted probabilty (False)
            light_weighted (bool): whether to return light-weighted (True) or
                mass-weighted (False) quantity
            v_edg (array): array of velocity-bin edges

        Returns:
            array: the desired distribution. If `collapse_cmps=True`, then array
                dimensions correspond to the order they are labelled in
                `which_dist` string (with 2 for 2D posiion x). If
                `collapse_cmps=False` the first array dimension represents
                different components.

        """
        assert which_dist in ['t',
                              'x_t',
                              'tx',
                              'z_tx',
                              'txz',
                              'x',
                              'z',
                              'tz_x',
                              'tz',
                              'tvxz',
                              'vx',
                              'v_x',
                              'v']
        is_conditional = '_' in which_dist
        if light_weighted is False:
            if is_conditional:
                p = self.get_conditional_mass_weighted_distributions(
                    which_dist,
                    density=density,
                    v_edg=v_edg)
            else:
                count = 0
                for i, (cmp, w) in enumerate(zip(self.component_list,self.weights)):
                    p_func = getattr(cmp, 'get_p_'+which_dist)
                    if 'v' in which_dist:
                        pi = w * p_func(v_edg=v_edg,
                                        density=density,
                                        light_weighted=False)
                    else:
                        pi = w * p_func(density=density, light_weighted=False)
                    if count == 0:
                        p = np.zeros((self.n_cmps,) + pi.shape)
                    p[i] = pi
                    count += 1
        else:
            p = self.get_light_weighted_distributions(
                which_dist,
                density=density,
                v_edg=v_edg)
        if collapse_cmps:
            p = np.sum(p, 0)
        return p

    def get_conditional_mass_weighted_distributions(self,
                                                    which_dist,
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
         # get an alphatized string for the joint distribution
        joint = ''.join(sorted(dist+marginal))
        p_joint = self.get_p(joint, density=False, v_edg=v_edg, collapse_cmps=False)
        p_marginal = self.get_p(marginal, density=False, v_edg=v_edg, collapse_cmps=True)
        # if x is in joint/marginalal, repalace it with xy to account for the
        # fact that x stands for 2D positon (x,y)
        joint = joint.replace('x', 'xy')
        marginal = marginal.replace('x', 'xy')
        # first dimension in the joint corresponts to component index:
        joint = 'i' + joint
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
                                 collapse_cmps=True,
                                 v_edg=None):
        """Construct volume element for converting densities to probabilties

        Args:
            which_dist (string): which density to evaluate
            collapse_cmps (bool): whether to collapse component densities
                together (True) or leave them in-tact (False)
            v_edg (array): array of velocity-bin edges

        Returns:
            array: The volume element with correct shape for `which_dist`

        """
        dist_string, marginal_string = which_dist.split('_')
        dist_string = dist_string.replace('x', 'xy')
        marginal_string = marginal_string.replace('x', 'xy')
        if collapse_cmps is True:
            count = 0
            ndim = len(dist_string)
        else:
            # first dimension is discrete indicator over components
            count = 1
            ndim = len(dist_string) + 1
        dvol = np.ones([1 for i in range(ndim)])
        na = np.newaxis
        slc = slice(0,None)
        for var in dist_string:
            if var=='t':
                da = self.ssps.delta_t
            elif var=='v':
                da = v_edg[1:] - v_edg[:-1]
            elif var=='x':
                da = np.array([self.dx])
            elif var=='y':
                da = np.array([self.dy])
            elif var=='z':
                da = self.ssps.delta_z
            idx = tuple([na for i in range(count)])
            idx = idx + (slc,)
            idx = idx + tuple([na for i in range(ndim-count-1)])
            dvol = dvol * da[idx]
            count += 1
        idx = tuple([slc for i in range(dvol.ndim)])
        idx = idx + tuple([na for i in range(len(marginal_string))])
        dvol = dvol[idx]
        return dvol

    def get_light_weighted_distributions(self,
                                         which_dist,
                                         density=True,
                                         v_edg=None):
        """Get light-weighted distributions

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
        dt = self.ssps.delta_t
        dz = self.ssps.delta_z
        if v_edg is not None:
            dv = v_edg[1:] - v_edg[:-1]
        dx2 = self.dx*self.dy
        na = np.newaxis
        count = 0
        # evaluate mass-weighted probabilties over all (t,x,z) or (t,v,x,z) if
        # v is in the target distribution
        for i, (cmp, w) in enumerate(zip(self.component_list,self.weights)):
            if 'v' in which_dist:
                pi = w * cmp.get_p_tvxz(v_edg=v_edg,
                                        density=False,
                                        light_weighted=False)
            else:
                pi = w * cmp.get_p_txz(density=False, light_weighted=False)
            if count == 0:
                p = np.zeros((self.n_cmps,) + pi.shape)
            p[i] = pi
            count += 1
        # apply light-weighting
        if 'v' in which_dist:
            p *= self.ssps.light_weights[na,:,na,na,na,:]
        else:
            p *= self.ssps.light_weights[na,:,na,na,:]
        p /= np.sum(p)
        # at this point p = light-weighted P(i,t,v,x,z) / P(i,t,x,z) ...
        # ... if v is / isn't in `which_dist` (and i indexes over components)
        # now evaluate the desired marginal/conditional
        if which_dist=='t':
            p = np.sum(p, (2,3,4))
            if density:
                p /= dt[na,:]
        elif which_dist=='x_t':
            p_tx = np.sum(p, 4)
            p_xt = np.moveaxis(p_tx, 1, -1)
            p_t = np.sum(p, (0,2,3,4))      # this must be mrg. over i
            p = p_xt/p_t[na,na,na,:]
            if density:
                p /= dx2
        elif which_dist=='tx':
            p = np.sum(p, 4)
            if density:
                p /= dx2*dt[na,:,na,na]
        elif which_dist=='z_tx':
            p_tx = np.sum(p, (0,4))
            p_ztx = np.moveaxis(p, -1, 1)
            p = p_ztx/p_tx[na,na,:,:,:]
            if density:
                p /= dz[na,:,na,na,na]
        elif which_dist=='txz':
            pass
            if density:
                dvol = dt[:,na,na,na]*dx2*dz[na,na,na,:]
                p /= dvol
        elif which_dist=='x':
            p = np.sum(p, (1,4))
            if density:
                p /= dx2
        elif which_dist=='z':
            p = np.sum(p, (1,2,3))
            if density:
                p /= dz
        elif which_dist=='tz_x':
            p_tzx = np.moveaxis(p, -1, 2)
            p_x = np.sum(p, (0,1,4))
            p = p_tzx/p_x
            if density:
                dvol = dt[na,:,na,na,na] * dz[na,na,:,na,na]
                p /= dvol
        elif which_dist=='tz':
            p = np.sum(p, (2,3))
            if density:
                dvol = dt[na,:,na]*dz[na,na,:]
                p /= dvol
        elif which_dist=='tvxz':
            pass
            if density:
                dvol = dt[:,na,na,na,na]*dv[na,:,na,na,na]*dx2*dz[na,na,na,na,:]
                p = p/dvol
        elif which_dist=='vx':
            p = np.sum(p, (1,5))
            if density:
                p /= (dx2*dv[:,na,na])
        elif which_dist=='v_x':
            p_vx = np.sum(p, (1,5))
            p_x = np.sum(p, (0,1,2,5))
            p = p_vx/p_x[na,na,:,:]
            if density:
                p /= dv[na,:,na,na]
        elif which_dist=='v':
            p = np.sum(p, (1,3,4,5))
            if density:
                p /= dv[na,:]
        else:
            raise ValueErrror('Unknown value of which_dist')
        return p

    def get_E_v_x(self):
        """Get the mean of the mixture-galaxy velocity map

        Returns:
            array

        """
        mu_i = np.array([cmp.get_E_v_x() for cmp in self.component_list])
        p_x_i = np.array([cmp.get_p_x(density=True) for cmp in self.component_list])
        E_v_x = np.sum(self.weights * p_x_i.T * mu_i.T, -1).T
        return E_v_x

    def get_jth_central_moment_v_x(self, j):
        """Get j'th central moment of the mixture-galaxy velocity map

        Args:
            j (int): which moment

        Returns:
            array

        """
        mu = self.get_E_v_x()
        k = np.arange(0, j+1, 1)
        j_choose_k = special.comb(j, k)
        na = np.newaxis
        muj_v_x = np.zeros_like(mu)
        for cmp_i, w_i in zip(self.component_list, self.weights):
            mu_i = cmp_i.get_E_v_x()
            muk_i = np.array([cmp_i.get_jth_central_moment_v_x(k0) for k0 in k])
            delta_mu_to_the_j_minus_k = (mu_i - mu)[na,:,:]**(j-k)[:,na,na]
            muj_i = w_i * np.sum(delta_mu_to_the_j_minus_k * muk_i, 0)
            muj_v_x += muj_i
        return muj_v_x

    def get_variance_v_x(self):
        """Get variance of the mixture-galaxy velocity map

        Returns:
            array

        """
        var_v_x = self.get_jth_central_moment_v_x(2)
        return var_v_x

    def get_skewness_v_x(self):
        """Get skewness of the mixture-galaxy velocity map

        Returns:
            array

        """
        mu3_v_x = self.get_jth_central_moment_v_x(3)
        var_v_x = self.get_jth_central_moment_v_x(2)
        skewness_v_x = mu3_v_x/var_v_x**1.5
        return skewness_v_x

    def get_kurtosis_v_x(self):
        """Get kurtosis of the mixture-galaxy velocity map

        Returns:
            array

        """
        mu4_v_x = self.get_jth_central_moment_v_x(4)
        var_v_x = self.get_jth_central_moment_v_x(2)
        kurtosis_v_x = mu4_v_x/var_v_x**2.
        return kurtosis_v_x

    def imshow(self,
               img,
               ax=None,
               label_ax=True,
               colorbar=True,
               colorbar_label='',
               **kw_imshow):
        """Plot an image

        Wrapper around `plt.imshow` which correctly aligns and rotates the image

        Args:
            img : the 2D image array
            ax : a `matplotlib` `axis` object
            label_ax (bool): whether to include axes tick marks and labels
            colorbar (bool):  whether to include a colorbar
            colorbar_label (string): the colorbar label
            **kw_imshow (dict): extra keyword parameters passed to `plt.imshow`

        Returns:
            a `matplotlib` `AxesImage` object

        """
        img = np.flipud(img.T)
        kw_imshow0 = {'extent':self.xrng + self.yrng}
        kw_imshow0.update(kw_imshow)
        if ax is None:
            ax = plt.gca()
        img = ax.imshow(img, **kw_imshow0)
        if colorbar:
            cbar = plt.colorbar(img)
            cbar.set_label(colorbar_label)
        if label_ax is False:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
        return img
