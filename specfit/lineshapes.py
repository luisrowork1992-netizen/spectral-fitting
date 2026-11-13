"""Area-normalised spectral line shapes, parameterised by centre and FWHM."""

from __future__ import annotations

import numpy as np
from scipy.special import voigt_profile

__all__ = ["gaussian", "lorentzian", "voigt", "pseudo_voigt", "voigt_fwhm", "measure_fwhm", "SHAPES"]

_G = 2 * np.sqrt(2 * np.log(2))  # FWHM = _G * sigma


def gaussian(x, x0, fwhm, area=1.0):
    sigma = fwhm / _G
    return area * np.exp(-0.5 * ((np.asarray(x) - x0) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def lorentzian(x, x0, fwhm, area=1.0):
    g = fwhm / 2
    return area * g / (np.pi * ((np.asarray(x) - x0) ** 2 + g * g))


def voigt(x, x0, fwhm_g, fwhm_l, area=1.0):
    """Convolution of a Gaussian (FWHM ``fwhm_g``) and a Lorentzian (FWHM ``fwhm_l``)."""
    return area * voigt_profile(np.asarray(x) - x0, fwhm_g / _G, fwhm_l / 2)


def pseudo_voigt(x, x0, fwhm, eta, area=1.0):
    """Linear mix ``eta * Lorentzian + (1 - eta) * Gaussian`` with a shared FWHM."""
    return eta * lorentzian(x, x0, fwhm, area) + (1 - eta) * gaussian(x, x0, fwhm, area)


def voigt_fwhm(fwhm_g, fwhm_l):
    """Olivero-Longbothum approximation to the Voigt FWHM (accurate to about 0.02 %)."""
    return 0.5346 * fwhm_l + np.sqrt(0.2166 * fwhm_l**2 + fwhm_g**2)


def measure_fwhm(x, y):
    """Full width at half maximum of a single sampled peak (linear interpolation)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmax(y))
    half = y[i] / 2
    left = np.where(y[:i] < half)[0]
    right = np.where(y[i:] < half)[0]
    if left.size == 0 or right.size == 0:
        raise ValueError("peak is not fully contained in the data")
    l, r = left[-1], i + right[0]
    xl = np.interp(half, [y[l], y[l + 1]], [x[l], x[l + 1]])
    xr = np.interp(half, [y[r], y[r - 1]], [x[r], x[r - 1]])
    return float(xr - xl)


#: name -> (function, names of the shape parameters after x0)
SHAPES = {
    "gaussian": (gaussian, ("fwhm",)),
    "lorentzian": (lorentzian, ("fwhm",)),
    "voigt": (voigt, ("fwhm_g", "fwhm_l")),
    "pseudo_voigt": (pseudo_voigt, ("fwhm", "eta")),
}
