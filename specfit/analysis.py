"""Beer-Lambert analysis, baseline removal, peak fitting and spectral unmixing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, nnls
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks

from .lineshapes import SHAPES

__all__ = [
    "absorbance", "transmittance", "concentration",
    "nm_to_wavenumber", "wavenumber_to_nm", "nm_to_ev", "ev_to_nm",
    "als_baseline", "detect_peaks", "Peak", "FitResult", "fit_peaks", "unmix",
]

HC_EV_NM = 1239.841984  # h c in eV nm


# --- Beer-Lambert --------------------------------------------------------
def absorbance(transmittance_value):
    """Decadic absorbance ``A = -log10(T)``."""
    return -np.log10(np.asarray(transmittance_value, dtype=float))


def transmittance(absorbance_value):
    return 10.0 ** (-np.asarray(absorbance_value, dtype=float))


def concentration(A, epsilon, path_length=1.0):
    """Beer-Lambert ``c = A / (epsilon l)`` (epsilon in L/(mol cm), l in cm -> mol/L)."""
    return np.asarray(A, dtype=float) / (epsilon * path_length)


# --- unit conversions ----------------------------------------------------
def nm_to_wavenumber(nm):
    """Wavelength in nm -> wavenumber in cm^-1."""
    return 1e7 / np.asarray(nm, dtype=float)


def wavenumber_to_nm(cm1):
    return 1e7 / np.asarray(cm1, dtype=float)


def nm_to_ev(nm):
    return HC_EV_NM / np.asarray(nm, dtype=float)


def ev_to_nm(ev):
    return HC_EV_NM / np.asarray(ev, dtype=float)


# --- baseline ------------------------------------------------------------
def als_baseline(y, lam: float = 1e5, p: float = 0.01, n_iter: int = 15):
    """Asymmetric least-squares baseline (Eilers and Boelens, 2005).

    ``lam`` sets the smoothness; ``p`` (small) sets the asymmetry, so peaks above
    the baseline get little weight. The right ``lam`` grows steeply with the number
    of points and with the peak widths: for about 1000 points and broad bands,
    values of 1e8 to 1e9 are typical.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(n, n - 2))
    P = lam * D.dot(D.T)
    w = np.ones(n)
    z = y
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, n, n)
        z = spsolve((W + P).tocsc(), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def detect_peaks(x, y, prominence: float | None = None, **kwargs):
    """Peak positions (in x units) using ``scipy.signal.find_peaks``.

    By default ``prominence`` is 5 % of the signal range.
    """
    y = np.asarray(y, dtype=float)
    if prominence is None:
        prominence = 0.05 * (y.max() - y.min())
    idx, props = find_peaks(y, prominence=prominence, **kwargs)
    return np.asarray(x)[idx], props


# --- peak fitting --------------------------------------------------------
@dataclass
class Peak:
    shape: str
    x0: float
    area: float
    widths: dict = field(default_factory=dict)

    def evaluate(self, x):
        f, names = SHAPES[self.shape]
        return f(x, self.x0, *[self.widths[n] for n in names], area=self.area)


@dataclass
class FitResult:
    peaks: list
    baseline: tuple          # polynomial coefficients, highest power first
    residual_rms: float
    params: np.ndarray
    stderr: np.ndarray
    success: bool

    def model(self, x):
        x = np.asarray(x, dtype=float)
        return sum(p.evaluate(x) for p in self.peaks) + np.polyval(self.baseline, x)


def _unpack(theta, spec, n_base):
    peaks, k = [], 0
    for shape, _ in spec:
        names = SHAPES[shape][1]
        x0, area = theta[k], theta[k + 1]
        widths = dict(zip(names, theta[k + 2:k + 2 + len(names)]))
        peaks.append(Peak(shape, x0, area, widths))
        k += 2 + len(names)
    return peaks, tuple(theta[k:k + n_base])


def fit_peaks(x, y, guesses, baseline_order: int = 0) -> FitResult:
    """Least-squares fit of a sum of peaks plus a polynomial baseline.

    ``guesses`` is a list of ``(shape, {"x0":..., "area":..., <width names>...})``.
    Widths are kept positive and ``eta`` (pseudo-Voigt) is kept in [0, 1].
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    theta0, lo, hi = [], [], []
    span = x.max() - x.min()
    for shape, g in guesses:
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}")
        names = SHAPES[shape][1]
        theta0 += [g["x0"], g["area"]] + [g[n] for n in names]
        lo += [x.min(), -np.inf] + [0.0 if n == "eta" else 1e-9 * span for n in names]
        hi += [x.max(), np.inf] + [1.0 if n == "eta" else span for n in names]
    n_base = baseline_order + 1
    theta0 += [0.0] * n_base
    lo += [-np.inf] * n_base
    hi += [np.inf] * n_base

    def resid(theta):
        peaks, base = _unpack(theta, guesses, n_base)
        return sum(p.evaluate(x) for p in peaks) + np.polyval(base, x) - y

    sol = least_squares(resid, np.array(theta0, float), bounds=(lo, hi), x_scale="jac")
    dof = max(1, x.size - sol.x.size)
    s2 = 2 * sol.cost / dof
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        stderr = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        stderr = np.full(sol.x.size, np.nan)
    peaks, base = _unpack(sol.x, guesses, n_base)
    return FitResult(peaks, base, float(np.sqrt(s2)), sol.x, stderr, bool(sol.success))


# --- unmixing ------------------------------------------------------------
def unmix(components, mixture, nonnegative: bool = True):
    """Concentrations of reference spectra in a mixture (Beer-Lambert is linear).

    ``components`` has shape (n_components, n_points). Returns ``(c, residual_norm)``.
    """
    Aref = np.asarray(components, dtype=float).T
    b = np.asarray(mixture, dtype=float)
    if nonnegative:
        c, rnorm = nnls(Aref, b)
    else:
        c, res, *_ = np.linalg.lstsq(Aref, b, rcond=None)
        rnorm = float(np.linalg.norm(Aref @ c - b))
    return c, float(rnorm)
