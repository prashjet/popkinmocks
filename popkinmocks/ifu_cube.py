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
        """Add noise to signal `ybar` to give observed `yobs` for desired SNR

        SNR = standard deviation of `ybar/yobs`

        Args:
            snr : the desired signal-to-noise ratio

        """
        nrm = stats.norm(0.*self.ybar, self.ybar/snr)
        self.snr = snr
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
              *args,
              **kwargs):
        """Evaluate densities of the galaxy

        5D densities over: stellar age (t), 2D position (x), velocity (v), and
        metallicity (z). Which density to evaluate is specificed by `which_dist`
        which can be one of:
        - 't'
        - 'x_t',
        - 'tx'
        - 'z_tx',
        - 'txz',
        - 'x',
        - 'z',
        - 'tz_x',
        - 'tz',
        - 'v_tx',
        - 'tvxz'
        - 'v_x',
        - 'v'
        where the underscore (if-present) represents conditioning i.e. calling
        `get_p('tz_x')` will return p(t,z|x). This calls the `get_p...` methods
        of the galaxy's consistituent components (n.b. these must be defined!)

        Args:
            which_dist (string): which density to evaluate
            collapse_cmps (bool): whether to collapse component densities
            together (True) or leave them in-tact (False)
            *args : extra arguments passed to component 'get_p...' method,
            typically a velocity bin-edge array
            **kwargs (type): extra keyword arguments passed to component
            'get_p...' method, typically booleans `density`, or `light_weighted`

        Returns:
            array: the desired density. If `collapse_cmps=True`, then array
            dimensions correspond to the order they are labelled in `which_dist`
            string (with 2 for 2D posiion x). If `collapse_cmps=False` the first
            array dimension represents different components.

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
                              'v_tx',
                              'tvxz',
                              'v_x',
                              'v']
        count = 0
        for i, (cmp, w) in enumerate(zip(self.component_list, self.weights)):
            p_func = getattr(cmp, 'get_p_'+which_dist)
            pi = w * p_func(*args, **kwargs)
            if count==0:
                p = np.zeros((self.n_cmps,) + pi.shape)
            p[i] = pi
            count += 1
        if collapse_cmps:
            p = np.sum(p, 0)
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
            ax: a `matplotlib` `axis` object

        """
        img = np.flipud(img.T)
        kw_imshow0 = {'extent':self.xrng + self.yrng}
        kw_imshow0.update(kw_imshow)
        print(kw_imshow0)
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
        return ax
