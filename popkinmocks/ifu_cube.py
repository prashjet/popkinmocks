import numpy as np
import matplotlib.pyplot as plt


class IFUCube(object):
    """The integral field unit (IFU) datacube

    Args:
        ssps : SSP templates in a `pkm.model_grids.milesSSPs` object
        nx1 (int): number of pixels in x1
        nx2 (int): number of pixels in x2
        nv (int): number of velocity bins
        x1rng (tuple): start/end co-ordinates in x1 (arbitrary units)
        x2rng (tuple): start/end co-ordinates in x2 (arbitrary units)
        vrng (tuple): start/end velocities in km/s
        interp_kind (string): type of interpolation for SSPs default = 'cubic'

    """

    def __init__(
        self,
        ssps=None,
        nx1=300,
        nx2=299,
        nv=200,
        x1rng=(-1, 1),
        x2rng=(-1, 1),
        vrng=(-1000, 1000),
        interp_kind="cubic"
    ):
        self.ssps = ssps
        self.nx = nx1
        self.ny = nx2
        self.x1rng = x1rng
        self.x2rng = x2rng
        x_edge = np.linspace(*x1rng, nx1 + 1)
        self.dx = x_edge[1] - x_edge[0]
        self.x = x_edge[:-1] + self.dx / 2.0
        y_edge = np.linspace(*x2rng, nx2 + 1)
        self.dy = y_edge[1] - y_edge[0]
        self.y = y_edge[:-1] + self.dy / 2.0
        xx, yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.xx = xx
        self.yy = yy
        self.nv = nv
        self.v_edg = np.linspace(*vrng, nv + 1)
        dv = self.v_edg[1] - self.v_edg[0]
        self.ssps._logarithmically_resample(dv=dv, interp_kind=interp_kind)
        self.ssps._calculate_fourier_transform()
        self.ssps.get_light_weights()

    def construct_volume_element(self, which_dist):
        """Construct volume element for converting densities to probabilties

        Args:
            which_dist (string): which distribution to evaluate volume element
                for e.g. 'vz' returns dv*dz, 't_x' returns dt

        Returns:
            array: The volume element shaped compatibly with `which_dist`

        """
        dist_is_conditional = "_" in which_dist
        if dist_is_conditional:
            dist_string, marginal_string = which_dist.split("_")
            marginal_string = marginal_string.replace("x", "xy")
        else:
            dist_string = which_dist
        dist_string = dist_string.replace("x", "xy")
        count = 0
        ndim = len(dist_string)
        dvol = np.ones([1 for i in range(ndim)])
        na = np.newaxis
        slc = slice(0, None)
        for var in dist_string:
            if var == "t":
                da = self.ssps.delta_t
            elif var == "v":
                da = self.v_edg[1:] - self.v_edg[:-1]
            elif var == "x":
                da = np.array([self.dx])
            elif var == "y":
                da = np.array([self.dy])
            elif var == "z":
                da = self.ssps.delta_z
            idx = tuple([na for i in range(count)])
            idx = idx + (slc,)
            idx = idx + tuple([na for i in range(ndim - count - 1)])
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
            which_variable (string): one of (t, v, x1, x2, z)

        Returns:
            array: the discretisation values used for this variable

        """
        if which_variable == "t":
            var = self.ssps.par_cents[1]
        elif which_variable == "v":
            v_edg = self.v_edg
            var = (v_edg[:-1] + v_edg[1:]) / 2.0
        elif which_variable == "z":
            var = self.ssps.par_cents[0]
        elif which_variable == "x1":
            var = self.x
        elif which_variable == "x2":
            var = self.y
        else:
            raise ValueError(f"Unknown variable: {which_variable}")
        return var

    def get_variable_edges(self, which_variable):
        """Get discretization bin edges of the variable v, x1, x2, t or z.

        Args:
            which_variable (string): one of (t, v, x1, x2, z)

        Returns:
            array: the discretisation bin edges used for this variable

        """
        if which_variable == "t":
            edg = self.ssps.par_edges[1]
        elif which_variable == "v":
            edg = self.v_edg
        elif which_variable == "z":
            edg = self.ssps.par_edges[0]
        elif which_variable == "x1":
            edg = np.concatenate([self.x - self.dx / 2, [self.x[-1] + self.dx / 2]])
        elif which_variable == "x2":
            edg = np.concatenate([self.y - self.dy / 2, [self.y[-1] + self.dy / 2]])
        else:
            raise ValueError(f"Unknown variable: {which_variable}")
        return edg

    def get_variable_size(self, which_variable):
        """Get size of variable array

        Args:
            which_variable (string): one of (t, v, x1, x2, z)

        Returns:
            array: the numer of discretisation elements for this variable

        """
        n = len(self.get_variable_values(which_variable))
        return n

    def get_distribution_shape(self, which_dist):
        """Get shape of a distribution given its string

        Args:
            which_dist (string): a valid distribution string.

        Returns:
            array: the shape of an array representing this distribution

        """
        which_dist = which_dist.replace("_", "")
        shape = []
        for var in which_dist:
            if var == "x":
                nx1 = self.get_variable_size("x1")
                nx2 = self.get_variable_size("x2")
                shape = shape + [nx1, nx2]
            else:
                shape = shape + [self.get_variable_size(var)]
        shape = tuple(shape)
        return shape

    def get_axis_label(self, which_dist):
        """Get axis labels for plotting

        Args:
            which_variable (string): which variable, one of (t, v, x1, x2, z)

        Returns:
            string: to be used in `ax.set_label`

        """
        if which_dist == "x1":
            lab = "$x_1$"
        elif which_dist == "x2":
            lab = "$x_2$"
        elif which_dist == "t":
            lab = "$t$ [Gyr]"
        elif which_dist == "z":
            lab = "$z$ [M/H]"
        elif which_dist == "v":
            lab = "$v$ [km/s]"
        else:
            raise ValueError("Unknown `which_dist`")
        return lab

    def _get_ticks(self, which_variable):
        """Get tick positions and labels for plotting

        Args:
            which_variable (string): which variable, one of (t, v, x1, x2, z)

        Returns:
            misc: if which_variable in [t,z] return ticks set in SSPs else
                return the string 'default'

        """
        if which_variable == "x1":
            ticks = "default"
        elif which_variable == "x2":
            ticks = "default"
        elif which_variable == "t":
            tick_pos = self.ssps.img_t_ticks
            tick_lab = self.ssps.t_ticks
            ticks = (tick_pos, tick_lab)
        elif which_variable == "z":
            tick_pos = self.ssps.img_z_ticks
            tick_lab = self.ssps.z_ticks
            ticks = (tick_pos, tick_lab)
        elif which_variable == "v":
            ticks = "default"
        else:
            raise ValueError("Unknown `which_variable`")
        return ticks

    def get_variable_extent(self, which_variable):
        """Get outermost bin edges for a given variable

        Args:
            which_variable (string): which variable, one of (t, v, x1, x2, z)

        Returns:
            tuple: (start, end) values of variable, i.e. outer most bin edges

        """
        if which_variable == "x1":
            ext = self.x1rng
        elif which_variable == "x2":
            ext = self.x2rng
        elif which_variable == "t":
            edges = self.ssps.par_edges[1]
            ext = (edges[0], edges[-1])
        elif which_variable == "z":
            edges = self.ssps.par_edges[0]
            ext = (edges[0], edges[-1])
        elif which_variable == "v":
            ext = (self.v_edg[0], self.v_edg[-1])
        else:
            raise ValueError("Unknown `which_variable`")
        return ext

    def _get_image_extent(self, which_variable):
        """Get extent used for plotting images

        Similar to `self.get_variable_extent` except this returns (0,1) if the variable
        is in [t,z] as these need special treatment due to irregular
        discretization

        Returns:
            tuple: (start, end) to be used with `plt.imshow`

        """
        if which_variable in ["x1", "x2", "v"]:
            ext = self.get_variable_extent(which_variable)
        elif which_variable in ["t", "z"]:
            ext = (0, 1)
        else:
            raise ValueError("Unknown `which_variable`")
        return ext

    def imshow(
        self,
        img,
        ax=None,
        label_ax=True,
        colorbar=True,
        colorbar_label="",
        view=["x1", "x2"],
        **kwargs,
    ):
        """Wrapper around `plt.imshow` to orient image and label axes

        Args:
            img (array): 2D image to show
            ax (`matplotlib` axis): optional, axis to plot on
            label_ax (bool): whether to show tick marks and labels
            colorbar (bool):  whether to show colorbar
            colorbar_label (string): colorbar label
            view (list): list of two strings amongst ['t','v','x1','x2','z']
                representing the variables on the x- and y- axes of the image
            **kwargs: keyword arguments passed to `plt.imshow` (must not
                include `extent`)

        Returns:
            a `matplotlib` `AxesImage` object

        """
        if ax is None:
            ax = plt.gca()
        img = np.flipud(img.T)
        extent = self._get_image_extent(view[0])
        extent += self._get_image_extent(view[1])
        img = ax.imshow(img, extent=extent, **kwargs)
        if colorbar:
            cbar = plt.colorbar(img)
            cbar.set_label(colorbar_label)
        if label_ax is False:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            for i, xy in enumerate(["x", "y"]):
                ticks = self._get_ticks(view[i])
                if ticks != "default":
                    tick_pos, tick_labs = ticks
                    getattr(ax, f"set_{xy}ticks")(tick_pos)
                    getattr(ax, f"set_{xy}ticklabels")(tick_labs)
                ax_lab = self.get_axis_label(view[i])
                getattr(ax, f"set_{xy}label")(ax_lab)
        return ax

    def plot(self, which_var, arr, *args, ax=None, xspacing="physical", **kwargs):
        """Wrapper around `plt.plot` to plot 1D plots and label axes

        Args:
            which_var (string): x-axis variable, one of (t, v, x1, x2, z)
            arr (array): y-axis data to plot
            *args: any extra args passed to `plt.plot`
            ax (`matplotlib` axis): optional, axis to plot on
            xspacing (string, optional): `physical` or `discrete`. If
                `physical`, x-axis is spaced in physical units, otherwise
                with equally spaced points per discretization point
            **kwargs (dict): any extra kwargs passed to `plt.plot`

        Returns:
            a `matplotlib` `AxesImage` object

        """
        if ax is None:
            ax = plt.gca()
        if xspacing == "physical":
            x = self.get_variable_values(which_var)
            extent = "default"
            ticks = "default"
        elif xspacing == "discrete":
            x = np.linspace(0, 1, arr.size)
            extent = self.get_variable_extent(which_var)
            ticks = self._get_ticks(which_var)
        img = ax.plot(x, arr, *args, **kwargs)
        if ticks != "default":
            tick_pos, tick_labs = ticks
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labs)
        ax_lab = self.get_axis_label(which_var)
        ax.set_xlabel(ax_lab)
        return ax

    def plot_spectrum(self, arr, *args, ax=None, **kwargs):
        """Wrapper around `plt.plot` to plot spectra

        Automatically selects if sampled in log-wavelength or wavelength and
        chooses x-axis values appropriately

        Args:
            arr (array): y-axis data to plot
            *args: any extra
            ax (`matplotlib` axis): optional, axis to plot on
            **kwargs (dict): extra keyword parameters passed to `plt.plot`

        Returns:
            a `matplotlib` `AxesImage` object

        """
        if ax is None:
            ax = plt.gca()
        arr = np.array(arr)
        if arr.shape[0] == self.ssps.lmd.size:
            wavelength = self.ssps.lmd
        elif arr.shape[0] == self.ssps.w.size:
            wavelength = np.exp(self.ssps.w)
        else:
            nlmd = self.ssps.lmd.size
            nw = self.ssps.w.size
            warning = "arr incorrect shape: first dimension should have size "
            warning += f"{nlmd} if sampled in wavelength or "
            warning += f"{nw} if sampled in log wavelength."
            print(warning)
        ax.plot(wavelength, arr, *args, **kwargs)
        ax.set_xlabel("Wavelength [$\AA$]")
        return ax
