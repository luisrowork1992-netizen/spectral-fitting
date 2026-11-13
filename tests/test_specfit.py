import numpy as np
import pytest
from scipy.integrate import quad

import specfit as sf

X = np.linspace(-50, 50, 200001)


@pytest.mark.parametrize("f, args", [
    (sf.gaussian, (2.0, 3.0)),
    (sf.lorentzian, (2.0, 3.0)),
    (sf.voigt, (2.0, 3.0, 1.5)),
    (sf.pseudo_voigt, (2.0, 3.0, 0.4)),
])
def test_line_shapes_are_area_normalised(f, args):
    # integrate over the whole real line: Lorentzian tails decay only as 1/x^2
    area, _ = quad(lambda x: f(x, *args, area=2.5), -np.inf, np.inf, limit=500)
    assert area == pytest.approx(2.5, rel=1e-6)


def test_lorentzian_area_with_its_long_tails():
    # the Lorentzian tails beyond |x| = 50 hold (2/pi) * atan-complement of the area
    inside = np.trapezoid(sf.lorentzian(X, 0.0, 3.0), X)
    expected = (2 / np.pi) * np.arctan(50 / 1.5)
    assert inside == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("fwhm", [0.5, 2.0, 7.0])
def test_fwhm_of_gaussian_and_lorentzian(fwhm):
    assert sf.measure_fwhm(X, sf.gaussian(X, 1.0, fwhm)) == pytest.approx(fwhm, rel=1e-4)
    assert sf.measure_fwhm(X, sf.lorentzian(X, 1.0, fwhm)) == pytest.approx(fwhm, rel=1e-4)


def test_voigt_limits():
    np.testing.assert_allclose(sf.voigt(X[::100], 0.0, 2.0, 1e-10), sf.gaussian(X[::100], 0.0, 2.0), atol=1e-8)
    np.testing.assert_allclose(sf.voigt(X[::100], 0.0, 1e-8, 2.0), sf.lorentzian(X[::100], 0.0, 2.0), atol=1e-6)


@pytest.mark.parametrize("fg, fl", [(1.0, 0.1), (1.0, 1.0), (0.3, 2.0), (2.0, 5.0)])
def test_olivero_voigt_width(fg, fl):
    measured = sf.measure_fwhm(X, sf.voigt(X, 0.0, fg, fl))
    assert sf.voigt_fwhm(fg, fl) == pytest.approx(measured, rel=3e-4)


def test_beer_lambert_round_trip():
    eps, l, c = 1.2e4, 1.0, 3.5e-5
    A = eps * l * c
    T = sf.transmittance(A)
    assert sf.absorbance(T) == pytest.approx(A)
    assert sf.concentration(sf.absorbance(T), eps, l) == pytest.approx(c)
    assert sf.transmittance(1.0) == pytest.approx(0.1)


def test_unit_conversions():
    assert sf.nm_to_wavenumber(500.0) == pytest.approx(20000.0)
    assert sf.wavenumber_to_nm(20000.0) == pytest.approx(500.0)
    assert sf.nm_to_ev(500.0) == pytest.approx(2.4797, abs=1e-4)
    assert sf.ev_to_nm(sf.nm_to_ev(633.0)) == pytest.approx(633.0)


def _synthetic(seed=0, noise=0.002):
    rng = np.random.default_rng(seed)
    x = np.linspace(400, 700, 1201)
    truth = [
        ("gaussian", dict(x0=480.0, area=12.0, fwhm=25.0)),
        ("lorentzian", dict(x0=540.0, area=8.0, fwhm=12.0)),
        ("voigt", dict(x0=610.0, area=10.0, fwhm_g=15.0, fwhm_l=6.0)),
    ]
    y = sum(sf.SHAPES[s][0](x, p["x0"], *[p[n] for n in sf.SHAPES[s][1]], area=p["area"]) for s, p in truth)
    baseline = 0.02 + 5e-5 * (x - 400)
    return x, y + baseline + noise * rng.standard_normal(x.size), truth, baseline


def test_fit_recovers_peak_parameters():
    x, y, truth, _ = _synthetic()
    guesses = []
    for s, p in truth:
        g = {k: v * (1.05 if k != "x0" else 1.0) for k, v in p.items()}
        g["x0"] = p["x0"] + 3.0
        guesses.append((s, g))
    res = sf.fit_peaks(x, y, guesses, baseline_order=1)
    assert res.success
    assert res.residual_rms == pytest.approx(0.002, rel=0.1)
    for peak, (s, p) in zip(res.peaks, truth):
        assert peak.x0 == pytest.approx(p["x0"], abs=0.2)
        assert peak.area == pytest.approx(p["area"], rel=0.02)
    np.testing.assert_allclose(res.baseline, [5e-5, 0.02 - 400 * 5e-5], atol=2e-3)
    assert np.all(np.isfinite(res.stderr))


def test_als_baseline_tracks_smooth_background():
    x, y, truth, baseline = _synthetic(noise=0.0)
    z = sf.als_baseline(y, lam=1e9, p=0.001)
    peak_height = (y - baseline).max()
    # the estimate stays within ~2 % of the tallest peak everywhere
    assert np.max(np.abs(z - baseline)) < 0.02 * peak_height
    # and the corrected spectrum keeps the peak areas
    assert np.trapezoid(y - z, x) == pytest.approx(30.0, rel=0.05)


def test_detect_peaks_finds_all_three():
    x, y, truth, _ = _synthetic()
    pos, _ = sf.detect_peaks(x, y)
    assert len(pos) == 3
    np.testing.assert_allclose(pos, [p["x0"] for _, p in truth], atol=1.0)


def test_unmixing_recovers_concentrations():
    x = np.linspace(400, 700, 301)
    refs = np.array([sf.gaussian(x, 450, 40), sf.gaussian(x, 550, 50), sf.lorentzian(x, 640, 30)])
    c_true = np.array([0.3, 1.2, 0.05])
    c, r = sf.unmix(refs, c_true @ refs)
    np.testing.assert_allclose(c, c_true, rtol=1e-9)
    assert r < 1e-10
    c2, _ = sf.unmix(refs, c_true @ refs, nonnegative=False)
    np.testing.assert_allclose(c2, c_true, rtol=1e-9)


def test_bad_shape_and_truncated_peak():
    with pytest.raises(ValueError):
        sf.fit_peaks([0, 1, 2], [0, 1, 0], [("triangle", {"x0": 1, "area": 1})])
    with pytest.raises(ValueError):
        sf.measure_fwhm(np.arange(10.0), np.arange(10.0))
