"""specfit: spectral line shapes, Beer-Lambert analysis, baselines, peak fitting and unmixing."""

from .analysis import (
    HC_EV_NM,
    FitResult,
    Peak,
    absorbance,
    als_baseline,
    concentration,
    detect_peaks,
    ev_to_nm,
    fit_peaks,
    nm_to_ev,
    nm_to_wavenumber,
    transmittance,
    unmix,
    wavenumber_to_nm,
)
from .lineshapes import SHAPES, gaussian, lorentzian, measure_fwhm, pseudo_voigt, voigt, voigt_fwhm

__version__ = "0.1.0"
