import numpy as np
from scipy import special, interpolate

import matplotlib.pyplot as plt
from . import read_miles


class modelGrid:
    def __init__(self, n, lmd_min, lmd_max, npars, par_dims, par_lims, X):
        self.n = n
        self.lmd_min = lmd_min
        self.lmd_max = lmd_max
        self.lmd = np.linspace(lmd_min, lmd_max, n)
        self.delta_lmd = (lmd_max - lmd_min) / n
        self.npars = npars
        self.par_dims = par_dims
        self.par_lims = par_lims
        self._check_par_input()
        self._get_par_grid()
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        self.X = X
        self.override_ticks = False

    def _check_par_input(self):
        check = len(self.par_dims) == self.npars
        if check is False:
            raise ValueError("par_dims is wrong length")
        check = np.array(self.par_lims).shape == (self.npars, 2)
        if check is False:
            raise ValueError("par_lims is wrong shape")
        if np.product(self.par_dims) > 1e5:
            raise ValueError("total size of par grid > 1e5")
        if self.par_mask_function is not None:
            # evaluate par_mask_function on zero-array to check input shape
            tmp = np.zeros((self.npars, 3))
            self.par_mask_function(tmp)
        pass

    def _get_par_grid(self, par_edges=None):
        # get param grid
        par_widths = []
        par_cents = []
        par_idx_arrays = []
        if par_edges is None:
            par_edges = []
            ziparr = zip(self.par_dims, self.par_lims)
            for i_p, (n_p, (lo, hi)) in enumerate(ziparr):
                edges = np.linspace(lo, hi, n_p + 1)
                par_edges += [edges]
        for edges in par_edges:
            par_widths += [edges[1:] - edges[0:-1]]
            par_cents += [(edges[0:-1] + edges[1:]) / 2.0]
            n_p = len(edges) - 1
            par_idx_arrays += [np.arange(n_p)]
        self.par_edges = par_edges
        self.par_widths = par_widths
        self.par_cents = par_cents
        par_grid_nd = np.meshgrid(*self.par_cents, indexing="ij")
        par_grid_nd = np.array(par_grid_nd)
        pars = par_grid_nd.reshape(self.npars, -1)
        par_idx = np.meshgrid(*par_idx_arrays, indexing="ij")
        par_idx = np.array(par_idx)
        par_idx = par_idx.reshape(self.npars, -1)
        # mask out entries
        if self.par_mask_function is not None:
            par_mask_nd = self.par_mask_function(par_grid_nd)
            par_mask_1d = self.par_mask_function(pars)
        else:
            par_mask_nd = np.zeros(self.par_dims, dtype=bool)
            par_mask_1d = np.zeros(np.product(self.par_dims), dtype=bool)
        self.par_mask_nd = par_mask_nd
        unmasked = par_mask_1d == False
        pars = pars[:, unmasked]
        par_idx = par_idx[:, unmasked]
        # store final arrays
        self.par_idx = par_idx
        self.pars = pars


class milesSSPs(modelGrid):
    """MILES SSP models

    MILES models (Vazdekis et al. 2010) with BaSTI isochrones for base
    alpha-content (Pietrinferni et al. 2004) and a Chabrier (2003) IMF.

    Args:
        lmd_min (float) : min wavelength (Angstrom)
        lmd_min (float) : max wavelength (Angstrom)
        age_lim (tuple) : (min, max) ages to keep (Gyr)
        z_lim (tuple) : (min, max) metallicities to keep [M/H]
        age_rebin (int) : downsampling factor for number of SSP ages
        z_rebin (int) : downsampling factor for number of SSP metallicities

    """

    def __init__(
        self, lmd_min=4700, lmd_max=6500, age_lim=None, z_lim=None, age_rebin=1, z_rebin=1
    ):
        ssps = read_miles.milesSSPs(
            mod_dir="MILES_BASTI_CH_baseFe",
            age_lim=age_lim,
            z_lim=z_lim,
            thin_age=1,
            thin_z=1,
        )
        ssps.truncate_wavelengths(lmd_min=lmd_min, lmd_max=lmd_max)
        n = ssps.X.shape[0]
        self.ssps = ssps
        npars = 2
        s = ("1) metallicity of stellar population", "2) age of stellar population")
        self.par_string = s
        pltsym = ["[Z/H]", "Age [Gyr]"]
        self.par_pltsym = pltsym
        self.n = n
        self.lmd_min = ssps.lmd[0]
        self.lmd_max = ssps.lmd[-1]
        self.lmd = ssps.lmd
        self.delta_lmd = (self.lmd_max - self.lmd_min) / (n - 1)
        self.npars = 2
        t_edg = self._get_par_edge_lims(ssps.t_unq)
        z_edg = self._get_par_edge_lims(ssps.z_unq)
        par_edges = [z_edg, t_edg]
        self.par_dims = (ssps.nz, ssps.nt)
        self.par_lims = ((0, 1), (0, 1))
        self.par_mask_function = None
        self._check_par_input()
        self._get_par_grid(par_edges=par_edges)
        p = self.pars.shape[-1]
        self.p = p
        self.min_n_p = np.min([n, p])
        self.X = ssps.X
        self.override_ticks = True
        self.set_tick_positions()
        tmp = self.par_edges[1]
        self.delta_t = tmp[1:] - tmp[:-1]
        tmp = self.par_edges[0]
        self.delta_z = tmp[1:] - tmp[:-1]
        self._rebin_par_grid(age_rebin=age_rebin, z_rebin=z_rebin)

    def _rebin_par_grid(self, age_rebin=1, z_rebin=1):
        new_z_edges = self.par_edges[0][::z_rebin]
        if new_z_edges[-1]!=self.par_edges[0][-1]:
            new_z_edges = np.concatenate((new_z_edges, [self.par_edges[0][-1]]))
        new_z_cents = (new_z_edges[1:] + new_z_edges[:-1])/2.
        new_delta_z = new_z_edges[1:] - new_z_edges[:-1]
        new_t_edges = self.par_edges[1][::age_rebin]
        if new_t_edges[-1]!=self.par_edges[1][-1]:
            new_t_edges = np.concatenate((new_t_edges, [self.par_edges[1][-1]]))
        new_t_cents = (new_t_edges[1:] + new_t_edges[:-1])/2.
        new_delta_t = new_t_edges[1:] - new_t_edges[:-1]
        new_par_cents = [new_z_cents, new_t_cents]
        new_par_edges = [new_z_edges, new_t_edges]
        new_par_dims = (new_z_cents.size, new_t_cents.size)
        X = self.X.reshape((-1,) + self.par_dims)
        new_X = np.zeros((X.shape[0],) + tuple(new_par_dims))
        na = np.newaxis
        dzdt = self.delta_z[:,na] * self.delta_t[na,:]
        Xdzdt = X*dzdt
        for i in range(new_par_dims[0]): # loop over z
           for j in range(new_par_dims[1]): # loop over t
                z0 = i*z_rebin
                if new_z_edges[i+1]==self.par_edges[0][-1]:
                    z1 = None
                else:
                   z1 = (i+1)*z_rebin
                z_slc = slice(z0,z1)
                t0 = j*age_rebin
                if new_t_edges[j+1]==self.par_edges[1][-1]:
                   t1 = None
                else:
                   t1 = (j+1)*age_rebin
                t_slc = slice(t0,t1)
                new_X[:,i,j] = np.sum(Xdzdt[:,z_slc,t_slc],(1,2))
                new_X[:,i,j] /= np.sum(dzdt[z_slc,t_slc])
        new_X = new_X.reshape((new_X.shape[0],) + (-1,))
        self.delta_t = new_delta_t
        self.delta_z = new_delta_z
        self.par_cents = new_par_cents
        self.par_edges = new_par_edges
        self.par_dims = new_par_dims
        self.X = new_X

    def _get_par_edge_lims(self, x_cnt):
        dx = x_cnt[1:] - x_cnt[:-1]
        x_edg_lo = x_cnt[1:] - dx / 2.0
        x_edg_hi = x_cnt[:-1] + dx / 2.0
        x_edg_lo = np.concatenate(([x_cnt[0] - dx[0] / 2.0], x_edg_lo))
        x_edg_hi = np.concatenate((x_edg_hi, [x_cnt[-1] + dx[-1] / 2.0]))
        assert np.allclose(x_edg_lo[1:], x_edg_hi[:-1])
        if np.abs(x_edg_lo[0]) < 0.011:
            x_edg_lo[0] = 0.0
        x_edg = np.concatenate((x_edg_lo, [x_edg_hi[-1]]))
        return x_edg

    def set_tick_positions(self, t_ticks=None, z_ticks=None):
        """Set tick positions for use in plotting

        These ticks positions are used when plotting SSPs variables (t,z) in
        discrete units when using `cube.plot(..., spacing='discrete')` or
        when using `cube.imshow`.

        Args:
            t_ticks (list) : ages where you would like a tick (Gyr)
            z_ticks (list) : metallicities where you'd like a tick [M/H]

        """
        if t_ticks is None and hasattr(self, "t_ticks") is False:
            t_ticks = [0.1, 1, 5, 13]
        elif t_ticks is None:
            t_ticks = self.t_ticks
        else:
            pass
        if z_ticks is None and hasattr(self, "z_ticks") is False:
            z_ticks = [-2, -1, 0]
        elif z_ticks is None:
            z_ticks = self.z_ticks
        else:
            pass
        # remove ticks which are out of bounds
        t_ticks = np.array(t_ticks)
        t_lims = self.par_edges[1]
        t_lo, t_hi = t_lims[0], t_lims[-1]
        idx = np.where((t_ticks > t_lo) & (t_ticks < t_hi))
        t_ticks = t_ticks[idx]
        z_ticks = np.array(z_ticks)
        z_lims = self.par_edges[0]
        z_lo, z_hi = z_lims[0], z_lims[-1]
        idx = np.where((z_ticks > z_lo) & (z_ticks < z_hi))
        z_ticks = z_ticks[idx]
        # go on with your day in peace
        self.t_ticks = t_ticks
        self.z_ticks = z_ticks
        tmp = np.linspace(0, 1, self.par_dims[1] + 1)
        self.img_t_ticks = np.interp(t_ticks, self.par_edges[1], tmp)
        tmp = np.linspace(0, 1, self.par_dims[0] + 1)
        self.img_z_ticks = np.interp(z_ticks, self.par_edges[0], tmp)

    def _logarithmically_resample(self, dv=5.0, interp_kind="cubic"):
        """Logarithmically resample the SSPs with given velocity spacing

        Interpolates the SSPs with even spacing in log-wavelength set by dv

        Args:
            dv (float) : desired velocity spacing (km/s)

        """
        speed_of_light = 299792.0
        dw = dv / speed_of_light
        w_in = np.log(self.lmd)
        f = interpolate.interp1d(
            w_in, self.X.T, kind=interp_kind, bounds_error=False, fill_value=0.0
        )
        w = np.arange(np.min(w_in), np.max(w_in), dw)
        Xw = f(w).T
        self.speed_of_light = speed_of_light
        self.dv = dv
        self.dw = dw
        self.w = w
        self.Xw = Xw

    def _calculate_fourier_transform(self, pad=False):
        """Get FFT of SSPs

        Args:
            pad : either False or string 'ppxf' to use the padding used in PPXF

        """
        self.n_pix = len(self.w)
        if pad == False:
            n_fft = len(self.w)
        elif pad == "ppxf":
            n_fft = 2 ** int(np.ceil(np.log2(self.Xw.shape[0])))
        else:
            raise ValueError("Unknown choice for pad")
        FXw = np.fft.rfft(self.Xw.T, n_fft)
        FXw = FXw.T
        self.pad = pad
        self.n_fft = n_fft
        self.FXw = FXw

    def get_light_weights(self):
        """Calculate light weights L(t,z) = int_lmd S(lmd;t,z) d lmd

        Sets the result to `self.light_weights`

        """
        light_weights = np.sum(self.X, 0)
        light_weights = np.reshape(light_weights, self.par_dims)
        # transpose (z,t) --> (t,z) to match convention used for components
        light_weights = light_weights.T
        self.light_weights = light_weights

    def get_ssp(self, ssp_id, spacing="wavelength"):
        """Get a single SSP model indexed by ssp_id

        Args:
            ssp_id (int): index of the SSP to be returned
            spacing (string, optional): either 'wavelength' or 'log-wavelength'

        """
        id_z, id_t = self.par_idx[:, ssp_id]
        z = self.par_cents[0][id_z]
        t = self.par_cents[1][id_t]
        if spacing == "wavelength":
            spectrum = self.X[:, ssp_id]
        elif spacing == "log-wavelength":
            spectrum = self.Xw[:, ssp_id]
        else:
            raise ValueError("Unknown choice for spacing")
        return t, z, spectrum


# end
