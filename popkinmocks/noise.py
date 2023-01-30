import numpy as np
from abc import abstractmethod
from scipy import stats


class Noise(object):

    def __init__(self, component):
        """Noise 

        Args:
            component : a `popkinmocks` component with pre-evaluated `ybar`

        """
        self.component = component

    def get_noisy_data(self, **kwargs):
        """yobs(x,w) = ybar(x,w) + eps(x,w)
        """
        ybar = self.component.ybar
        eps = self.sample(**kwargs)
        yobs = ybar + eps
        return yobs
    
    @abstractmethod
    def sample(self):
        pass


class Diagonal(Noise):

    def sample(self, **kwargs):
        """eps(x,w) ~ Norm(0 , sigma(x,w))

        Args:
            snr : the desired signal-to-noise ratio

        """
        sigma = self.get_sigma(**kwargs)
        mu = np.zeros_like(self.component.ybar)
        nrm = stats.norm(mu, sigma)
        return nrm.rvs()

    @abstractmethod
    def get_sigma(self):
        pass


class ConstantSNR(Diagonal):

    def get_sigma(self, snr=100.):
        """sigma(x,w) = ybar(x,w)/snr

        Args:
            snr : the desired constant signal-to-noise ratio

        """
        return self.component.ybar/snr

class ShotNoise(Diagonal):

    def get_sigma(self, snr=100.):
        """sigma(x,w) =  K ybar(x,w)^(1/2)

        Proportionality constant K chosen so that brightest pixel has desired
        snr i.e. you set the maximum snr

        Args:
            snr : the desired maximum signal-to-noise ratio in a pixel

        """
        max_ybar = np.max(self.component.ybar)
        K = max_ybar**0.5/snr
        sigma = K * self.component.ybar**0.5
        return sigma