import os
import numpy as np
from astropy.io import fits

from . import read_miles


class milesSSPs:
    def __init__(
        self,
        mod_dir="MILES_BASTI_CH_baseFe",
        age_lim=None,
        z_lim=None,
        thin_age=1,
        thin_z=1,
    ):
        if mod_dir == None:
            self.list_mod_directories()
            return
        self.data_dir = read_miles.__file__.replace("read_miles.py", "../data/")
        # Directory structure should be /data_dir/mod_dir/*.fits, where:
        # - mod_dir is a directory containing a grid of SSP models
        # - *.fits are fits files of model spetctra
        # in format provided by this webtool:
        # http://research.iac.es/proyecto/miles/pages/webtools/
        if mod_dir[-1] != "/":
            mod_dir = mod_dir + "/"
        self.mod_dir = mod_dir
        modfiles = os.listdir(self.data_dir + self.mod_dir)
        if "all_spectra.npy" in modfiles:
            modfiles.remove("all_spectra.npy")
        self.modfiles = modfiles
        self.split_modfiles()
        self.age_lim = age_lim
        self.z_lim = z_lim
        self.filter_age_metallicity_limits()
        self.thin_age = thin_age
        self.thin_z = thin_z
        self.thin_model_grid(thin_age=thin_age, thin_z=thin_z)
        self.check_age_metallicity_grid()
        self.get_lmd()
        self.read_spectra()

    def list_mod_directories(self):
        direcs = os.listdir(self.data_dir)
        print("Initilise with 'mod_dir' set to:")
        for i, d in enumerate(direcs):
            print(f"{i+1}) {d}")

    def split_modfiles(self):
        IMFstr = []
        zstr, z = [], []
        tstr, t = [], []
        endstr = []
        for mod in self.modfiles:
            tmp = mod.split("Z")
            IMFstr += [tmp[0]]
            tmp = tmp[1].split("T")
            if len(tmp) == 3:  # fix if basti isochrones used (i.e iT in name)
                tmp = [tmp[0], f"{tmp[1]}T{tmp[2]}"]
            zstr += [tmp[0]]
            tmp = tmp[1].split("_i")
            tstr += [tmp[0]]
            endstr += [t for t in tmp[1:]]
            if zstr[-1][0] == "m":
                z += [-float(zstr[-1][1:])]
            elif zstr[-1][0] == "p":
                z += [float(zstr[-1][1:])]
            else:
                raise ValueError("???")
            t += [float(tstr[-1])]
        self.IMFstr = np.array(IMFstr)
        self.zstr = np.array(zstr)
        self.z = np.array(z)
        self.tstr = np.array(tstr)
        self.t = np.array(t)
        self.endstr = np.array(endstr)

    def filter_age_metallicity_limits(self):
        if self.age_lim is None:
            age_lim = (-3, 30)
        else:
            age_lim = self.age_lim
        if self.z_lim is None:
            z_lim = (-3, 3)
        else:
            z_lim = self.z_lim
        idx_zt = np.where(
            (self.t >= age_lim[0])
            & (self.t <= age_lim[1])
            & (self.z >= z_lim[0])
            & (self.z <= z_lim[1])
        )
        self.IMFstr = self.IMFstr[idx_zt]
        self.zstr = self.zstr[idx_zt]
        self.z = self.z[idx_zt]
        self.tstr = self.tstr[idx_zt]
        self.t = self.t[idx_zt]
        self.endstr = self.endstr[idx_zt]
        self.modfiles = list(np.array(self.modfiles)[idx_zt])

    def thin_model_grid(self, thin_age=1, thin_z=1):
        t_thin = np.unique(self.t)[::thin_age]
        z_thin = np.unique(self.z)[::thin_z]
        idx_zt = np.where(np.isin(self.t, t_thin) & np.isin(self.z, z_thin))
        self.IMFstr = self.IMFstr[idx_zt]
        self.zstr = self.zstr[idx_zt]
        self.z = self.z[idx_zt]
        self.tstr = self.tstr[idx_zt]
        self.t = self.t[idx_zt]
        self.endstr = self.endstr[idx_zt]
        self.modfiles = list(np.array(self.modfiles)[idx_zt])

    def join_modfiles(self, z=0, t=0):
        idx = np.where((self.t == t) & (self.z == z))
        if idx[0].size == 1:
            idx = idx[0][0]
        else:
            raise ValueError("not exactly one file of given name")
        istr = self.IMFstr[idx]
        zstr = self.zstr[idx]
        tstr = self.tstr[idx]
        estr = self.endstr[idx]
        modfile = f"{istr}Z{zstr}T{tstr}_i{estr}"
        return modfile

    def check_age_metallicity_grid(self):
        z_unq, t_unq = np.unique(self.z), np.unique(self.t)
        self.nz = len(z_unq)
        self.nt = len(t_unq)
        if len(self.modfiles) != (self.nz * self.nt):
            raise ValueError("Not age metallicity grid")
        self.z_unq = z_unq
        self.t_unq = t_unq

    def read_spectrum(self, z=0, t=0):
        modfile = self.join_modfiles(t=t, z=z)
        spectrum = fits.open(self.data_dir + self.mod_dir + modfile)
        spectrum = spectrum[0].data
        return spectrum

    def read_spectra(self):
        c1 = (self.age_lim is None) and (self.z_lim is None)
        c2 = (self.thin_age == 1) and (self.thin_z == 1)
        allspecfile = self.data_dir + self.mod_dir + "all_spectra.npy"
        if c1 and c2:
            if os.path.isfile(allspecfile):
                self.X = np.load(allspecfile)
        else:
            s0 = self.read_spectrum(z=self.z_unq[0], t=self.t_unq[0])
            if len(s0) != len(self.lmd):
                raise ValueError("len(wavelength array) != len(spectrum)")
            n = len(s0)
            p = self.nz * self.nt
            X = np.zeros((n, p))
            cnt = 0
            for z0 in self.z_unq:
                for t0 in self.t_unq:
                    s0 = self.read_spectrum(z=z0, t=t0)
                    X[:, cnt] = s0
                    cnt += 1
            self.X = X
        if c1 and c2 and not os.path.isfile(allspecfile):
            np.save(allspecfile, self.X)

    def get_lmd(self):
        self.lmd = np.arange(3540.5, 7409.6 + 0.9, 0.9)

    def truncate_wavelengths(self, lmd_min=None, lmd_max=None):
        if lmd_min is None:
            lmd_min = np.min(self.lmd)
        if lmd_max is None:
            lmd_max = np.max(self.lmd)
        if lmd_min < np.min(self.lmd) or lmd_max > np.max(self.lmd):
            str = "Requested wavelength range outside of SSP limits:, {0}-{1}"
            str = str.format(np.min(self.lmd), np.max(self.lmd))
            raise ValueError(str)
        idx = np.where((self.lmd >= lmd_min) & (self.lmd <= lmd_max))
        self.X = self.X[idx[0], :]
        self.lmd = self.lmd[idx]

    def bin_pixels(self, pix_per_bin=1):
        if pix_per_bin == 1:
            pass
        else:
            pix_per_bin = int(pix_per_bin)
            n, p = self.X.shape
            r = n % pix_per_bin
            tmp = self.lmd[:-r]
            tmp = np.sum(np.reshape(tmp, (pix_per_bin, -1)), 0)
            self.lmd = tmp
            tmp = self.X[:-r, :]
            self.X = np.sum(np.reshape(tmp, (pix_per_bin, -1, p)), 0)

    def reset(self):
        self.get_lmd()
        self.read_spectra()


# end
