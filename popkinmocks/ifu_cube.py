import numpy as np
from scipy import stats, interpolate, optimize, special
import matplotlib.pyplot as plt
import os
import dill

class IFUCube(object):
    """The integral field unit (IFU) datacube

    Args:
        ssps : SSP templates in a `pkm.model_grids.milesSSPs` object
        nx (int): number of pixels in x-dimension
        ny (int): number of pixels in y-dimension
        xrng (tuple): start/end co-ordinates in x-direction
        yrng (tuple): start/end co-ordinates in x-direction
        v_edg (array): velocity bin edges

    """
    def __init__(self,
                 ssps=None,
                 nx=300,
                 ny=299,
                 xrng=(-1,1),
                 yrng=(-1,1),
                 v_edg=np.linspace(-1000, 1000, 201)):
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
        self.v_edg = v_edg

    def add_noise(self, snr=100.):
        """Add noise with SNR which is constant with position and wavelength

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
        self.noise_constant = noise_constant
        nrm = stats.norm(0, self.noise_constant*self.ybar**0.5)
        self.noise = nrm.rvs()
        self.yobs = self.ybar + self.noise

    def save_data(self,
                  direc=None,
                  fname=None):
        """Save the IFUCube object as a dill dump

        Args:
            direc (string): directory name
            fname (string): filename

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        with open(direc + fname, 'wb') as f:
            dill.dump(self, f)

    def save_numpy(self,
                   direc=None,
                   fname=None):
        """Save the IFUCube object as a numpy npz file

        Args:
            direc (string): directory name
            fname (string): filename

        """
        if os.path.isdir(direc) is False:
            os.mkdir(direc)
        v_edg = self.v_edg
        u_edg = np.log(1. + v_edg/self.ssps.speed_of_light)
        p_tvxz = self.get_p('tvxz', collapse_cmps=True, density=True)
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
                da = self.ssps.delta_t
            elif var=='v':
                da = self.v_edg[1:] - self.v_edg[:-1]
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
        if dist_is_conditional:
            idx = idx + tuple([na for i in range(len(marginal_string))])
        dvol = dvol[idx]
        return dvol

    def get_variable_values(self, which_variable):
        """Get values of the variable v, x1, x2, t or z.

        Args:
            which_variable (string): which variable, one of v, x1, x2, t or z.

        Returns:
            array: the discretisation values used for this variable

        """
        if which_variable == 't':
            var = self.ssps.par_cents[1]
        elif which_variable == 'v':
            v_edg = self.v_edg
            var = (v_edg[:-1] + v_edg[1:])/2.
        elif which_variable == 'z':
            var = self.ssps.par_cents[0]
        elif which_variable == 'x1':
            var = self.x
        elif which_variable == 'x2':
            var = self.y
        else:
            raise ValueError(f'Unknown variable: {which_variable}')
        return var

    def get_variable_size(self, which_variable):
        """Get size of variable array

        Args:
            which_variable (string): which variable, one of v, x1, x2, t or z.

        Returns:
            array: the discretisation values used for this variable

        """
        n = len(self.get_variable_values(which_variable))
        return n

    def get_distribution_shape(self, which_dist):
        """Get values of the variable v, x1, x2, t or z.

        Args:
            which_variable (string): which variable, one of v, x1, x2, t or z.

        Returns:
            array: the discretisation values used for this variable

        """
        which_dist = which_dist.replace('_', '')
        shape = []
        for var in which_dist:
            if var == 'x':
                nx1 = self.get_variable_size('x1')
                nx2 = self.get_variable_size('x2')
                shape = shape + [nx1, nx2]
            else:
                shape = shape + [self.get_variable_size(var)]
        shape = tuple(shape)
        return shape

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
