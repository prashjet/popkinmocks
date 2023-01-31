import numpy as np
from scipy import special, interpolate

import matplotlib.pyplot as plt
from . import read_miles

class modelGrid:

    def __init__(self,
                 n,
                 lmd_min,
                 lmd_max,
                 npars,
                 par_dims,
                 par_lims,
                 par_mask_function,
                 normalise_models=False,    # False / column / volume
                 X=None):
        self.n = n
        self.lmd_min = lmd_min
        self.lmd_max = lmd_max
        self.lmd = np.linspace(lmd_min, lmd_max, n)
        self.delta_lmd = (lmd_max-lmd_min)/n
        self.npars = npars
        self.par_dims = par_dims
        self.par_lims = par_lims
        self.par_mask_function = par_mask_function
        self.check_par_input()
        self.get_par_grid()
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        if X==None:
            self.X = self.get_X(self.lmd[:, np.newaxis], self.pars)
        if normalise_models is not False:
            self.normalise(normalise_models)
        self.override_ticks = False

    def add_constant(self, c):
        self.X += c

    def normalise(self, normalise_models):
        if normalise_models==False:
            pass
        elif normalise_models=='column':
            self.X = self.X/self.delta_lmd/np.sum(self.X, 0)
        elif normalise_models=='volume':
            par_widths_nd = np.meshgrid(*self.par_widths, indexing='ij')
            par_vols = np.product(np.array(par_widths_nd), axis=0)
            par_vols = np.ravel(par_vols)
            self.par_vols = par_vols
            self.X *= par_vols
        else:
            raise ValueError('unkown normalise option')

    def check_par_input(self):
        check = (len(self.par_dims) == self.npars)
        if check is False:
            raise ValueError('par_dims is wrong length')
        check = (np.array(self.par_lims).shape == (self.npars, 2))
        if check is False:
            raise ValueError('par_lims is wrong shape')
        if np.product(self.par_dims) > 1e5:
            raise ValueError('total size of par grid > 1e5')
        if self.par_mask_function is not None:
            # evaluate par_mask_function on zero-array to check input shape
            tmp = np.zeros((self.npars, 3))
            self.par_mask_function(tmp)
        pass

    def par_description(self):
        print(self.par_string)

    def get_par_grid(self, par_edges=None):
        # get param grid
        par_widths = []
        par_cents = []
        par_idx_arrays = []
        if par_edges is None:
            par_edges = []
            ziparr = zip(self.par_dims, self.par_lims)
            for i_p, (n_p, (lo, hi)) in enumerate(ziparr):
                edges = np.linspace(lo, hi, n_p+1)
                par_edges += [edges]
        for edges in par_edges:
            par_widths += [edges[1:] - edges[0:-1]]
            par_cents += [(edges[0:-1] + edges[1:])/2.]
            n_p = len(edges) - 1
            par_idx_arrays += [np.arange(n_p)]
        self.par_edges = par_edges
        self.par_widths = par_widths
        self.par_cents = par_cents
        par_grid_nd = np.meshgrid(*self.par_cents, indexing='ij')
        par_grid_nd = np.array(par_grid_nd)
        pars = par_grid_nd.reshape(self.npars, -1)
        par_idx = np.meshgrid(*par_idx_arrays, indexing='ij')
        par_idx = np.array(par_idx)
        par_idx = par_idx.reshape(self.npars, -1)
        # mask out entries
        if self.par_mask_function is not None:
            par_mask_nd = self.par_mask_function(par_grid_nd)
            par_mask_1d = self.par_mask_function(pars)
        else:
            par_mask_nd = np.zeros(self.par_dims, dtype=bool)
            par_mask_1d = np.zeros(np.product(self.par_dims),
                                   dtype=bool)
        self.par_mask_nd = par_mask_nd
        unmasked = par_mask_1d==False
        pars = pars[:, unmasked]
        par_idx = par_idx[:, unmasked]
        # store final arrays
        self.par_idx = par_idx
        self.pars = pars

    def get_X(self, lmd, pars):
        """Placeholder for get_X"""
        pass

    def plot_models(self,
                    ax=None,
                    kw_plot={'color':'k',
                             'alpha':0.1}):
        if ax==None:
            ax = plt.gca()
        ax.plot(self.lmd, self.X, **kw_plot)
        ax.set_xlabel('$\lambda$')
        plt.show()

    def reshape_beta(self, beta):
        shape = beta.shape
        if shape[-1]!=self.p:
            raise ValueError('beta must have shape (..., {0})'.format(self.p))
        dims = tuple(shape[0:-1]) + tuple(self.par_dims)
        beta_reshape = np.zeros(dims)
        idx = tuple([slice(0,None) for i in range(len(beta.shape)-1)])
        idx += tuple([self.par_idx[i] for i in range(self.npars)])
        beta_reshape[idx] = beta
        return beta_reshape

    def get_finite_difference_tikhonov_matrix(self,
                                              axis=None,
                                              n=0,
                                              ax_weights=None):

        # the axes over which we want to regularise
        if axis is None:
            axis = list(range(self.npars))
        n_ax = len(axis)

        # n = order of regularisation
        # i.e. we minimise the sum of squared n'th derivs
        if type(n) is int:
            n = n * np.ones(n_ax, dtype=int)
        else:
            n = np.array(n, dtype=int)
            assert len(n)==n_ax

        # relative weight of regularisation in each axis
        if ax_weights is None:
            ax_weights = np.ones(n_ax)

        # make C distance array
        # C[n, i, j] = signed distance in n'th parameter between pixel i and j
        shape = (self.npars, self.p, self.p)
        C = np.full(shape, self.p+1, dtype=int)
        for i in range(self.npars):
            pi = self.par_idx[i]
            Ci = pi - pi[:, np.newaxis]
            C[i, :, :] = Ci

        # make D distance arary
        # D[n, i, j] = { C[n, i, j] if other {1,...N}-{n} parameters are equal
        #              { self.p+1 otherwise (this won't interfere with T calc.)
        shape = (len(axis), self.p, self.p)
        D = np.full(shape, self.p+1, dtype=int)
        all_axes = set(range(self.npars))
        for i, ax0 in enumerate(axis):
            other_axes = all_axes - set([ax0])
            slice = np.sum(np.abs(C[tuple(other_axes), :, :]), 0)
            slice = np.where(slice==0)
            D[i][slice] = C[ax0][slice]

        # make tikhanov matrix
        shape = (len(axis), self.p, self.p)
        T = np.zeros(shape, dtype=int)
        for i, n0 in enumerate(n):
            k = np.arange(n0+1)
            d = int(np.ceil(n0/2.)) - k
            t = (-1)**k * special.binom(n0, k)
            for d0, t0 in zip(d, t):
                T[i][D[i]==d0] = t0

        # collapse along axes
        T = np.sum(T.T * ax_weights, -1).T

        return T

class milesSSPs(modelGrid):

    def __init__(self,
                 miles_mod_directory='MILES_BASTI_CH_baseFe',
                 pix_per_bin=1,
                 lmd_min=4700,
                 lmd_max=6500,
                 age_lim=None,
                 z_lim=None,
                 thin_age=1,
                 thin_z=1,
                 normalise_models=False):

        ssps = read_miles.milesSSPs(mod_dir=miles_mod_directory,
                                    age_lim=age_lim,
                                    z_lim=z_lim,
                                    thin_age=thin_age,
                                    thin_z=thin_z)
        ssps.truncate_wavelengths(lmd_min=lmd_min, lmd_max=lmd_max)
        ssps.bin_pixels(pix_per_bin=pix_per_bin)
        n = ssps.X.shape[0]
        self.ssps = ssps
        npars = 2
        s = ("1) metallicity of stellar population",
             "2) age of stellar population")
        self.par_string = s
        pltsym = ['[Z/H]', 'Age [Gyr]']
        self.par_pltsym = pltsym
        self.n = n
        self.lmd_min = ssps.lmd[0]
        self.lmd_max = ssps.lmd[-1]
        self.lmd = ssps.lmd
        self.delta_lmd = (self.lmd_max-self.lmd_min)/(n-1)
        self.npars = 2
        t_edg = self.get_par_edge_lims(ssps.t_unq)
        z_edg = self.get_par_edge_lims(ssps.z_unq)
        par_edges = [z_edg, t_edg]
        self.par_dims = (ssps.nz, ssps.nt)
        self.par_lims = ((0,1), (0,1))
        self.par_mask_function = None
        self.check_par_input()
        self.get_par_grid(par_edges=par_edges)
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        self.X = ssps.X
        # remaining pars from modgrid
        if normalise_models is not False:
            self.normalise(normalise_models)
        self.override_ticks = True
        self.set_tick_positions()
        tmp = self.par_edges[1]
        self.delta_t = tmp[1:] - tmp[:-1]
        tmp = self.par_edges[0]
        self.delta_z = tmp[1:] - tmp[:-1]

    def get_par_edge_lims(self, x_cnt):
        dx = x_cnt[1:] - x_cnt[:-1]
        x_edg_lo = x_cnt[1:] - dx/2.
        x_edg_hi = x_cnt[:-1] + dx/2.
        x_edg_lo = np.concatenate(([x_cnt[0]-dx[0]/2.], x_edg_lo))
        x_edg_hi = np.concatenate((x_edg_hi, [x_cnt[-1]+dx[-1]/2.]))
        assert np.allclose(x_edg_lo[1:], x_edg_hi[:-1])
        if np.abs(x_edg_lo[0])<0.011:
            x_edg_lo[0] = 0.
        x_edg = np.concatenate((x_edg_lo, [x_edg_hi[-1]]))
        return x_edg

    def set_tick_positions(self,
                           t_ticks=None,
                           z_ticks=None):
        if t_ticks is None and hasattr(self, 't_ticks') is False:
            t_ticks = [0.1, 1, 5, 13]
        elif t_ticks is None:
            t_ticks = self.t_ticks
        else:
            pass
        if z_ticks is None and hasattr(self, 'z_ticks') is False:
            z_ticks = [-2, -1, 0]
        elif z_ticks is None:
            z_ticks = self.z_ticks
        else:
            pass
        # remove ticks which are out of bounds
        t_ticks = np.array(t_ticks)
        t_lims = self.par_edges[1]
        t_lo, t_hi = t_lims[0], t_lims[-1]
        idx = np.where((t_ticks>t_lo) & (t_ticks<t_hi))
        t_ticks = t_ticks[idx]
        z_ticks = np.array(z_ticks)
        z_lims = self.par_edges[0]
        z_lo, z_hi = z_lims[0], z_lims[-1]
        idx = np.where((z_ticks>z_lo) & (z_ticks<z_hi))
        z_ticks = z_ticks[idx]
        # go on with your day in peace
        self.t_ticks = t_ticks
        self.z_ticks = z_ticks
        tmp = np.linspace(0, 1, self.par_dims[1]+1)
        self.img_t_ticks = np.interp(t_ticks,
                                     self.par_edges[1],
                                     tmp)
        tmp = np.linspace(0, 1, self.par_dims[0]+1)
        self.img_z_ticks = np.interp(z_ticks,
                                     self.par_edges[0],
                                     tmp)

    def logarithmically_resample(self, dv=5.):
        speed_of_light = 299792.
        dw = dv/speed_of_light
        w_in = np.log(self.lmd)
        f = interpolate.interp1d(w_in,
                                 self.X.T,
                                 kind='cubic',
                                 bounds_error=False,
                                 fill_value=0.)
        w = np.arange(np.min(w_in), np.max(w_in), dw)
        Xw = f(w).T
        self.speed_of_light = speed_of_light
        self.dv = dv
        self.dw = dw
        self.w = w
        self.Xw = Xw

    def calculate_fourier_transform(self, pad=False):
        self.n_pix = len(self.w)
        if pad==False:
            n_fft = len(self.w)
        elif pad=='ppxf':
            n_fft = 2**int(np.ceil(np.log2(self.Xw.shape[0])))
        else:
            raise ValueError('Unknown choice for pad')
        FXw = np.fft.rfft(self.Xw.T, n_fft)
        FXw = FXw.T
        self.pad = pad
        self.n_fft = n_fft
        self.FXw = FXw

    def normalise_to_median(self):
        self.X /= np.median(self.X)

    def get_light_weights(self):
        light_weights = np.sum(self.X, 0)
        light_weights = np.reshape(light_weights, self.par_dims)
        # transpose (z,t) --> (t,z) to match convention used for components
        light_weights = light_weights.T
        self.light_weights = light_weights

    def get_ssp_wavelength_spacing(self, ssp_id):
        id_z, id_t = self.par_idx[:,ssp_id]
        z = self.par_cents[0][id_z]
        t = self.par_cents[1][id_t]
        spectrum = self.X[:,ssp_id]
        return t, z, spectrum

    def get_ssp_log_wavelength_spacing(self, ssp_id):
        id_z, id_t = self.par_idx[:,ssp_id]
        z = self.par_cents[0][id_z]
        t = self.par_cents[1][id_t]
        spectrum = self.Xw[:,ssp_id]
        return t, z, spectrum


# end
