# spectral-fitting

![tests](https://github.com/luisrowork1992-netizen/spectral-fitting/actions/workflows/tests.yml/badge.svg)

`specfit` is a compact toolkit for **optical spectroscopy data**: UV-Vis absorption, fluorescence, Raman and IR. It provides area-normalised line shapes, Beer–Lambert analysis, baseline removal, peak detection, least-squares peak fitting with uncertainties, and multicomponent unmixing.

## Features

- Line shapes parameterised by centre, FWHM and area, so the fitted area is the integrated intensity:
  - Gaussian
  - Lorentzian
  - Voigt (exact, using `scipy.special.voigt_profile`)
  - pseudo-Voigt
- Olivero–Longbothum Voigt width formula, plus a numerical FWHM measurement
- Beer–Lambert helpers: absorbance ↔ transmittance, and concentration from εl
- Unit conversions: nm ↔ cm⁻¹ ↔ eV
- Asymmetric-least-squares (ALS) baseline, following Eilers and Boelens
- Peak detection by prominence
- `fit_peaks`: any mix of shapes plus a polynomial baseline
  - bounded parameters
  - standard errors from the Jacobian
- `unmix`: non-negative least-squares concentrations of reference spectra in a mixture

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/spectral-fitting.git
cd spectral-fitting
pip install -e .
```

## Quick start

```python
import numpy as np
import specfit as sf

x = np.linspace(400, 700, 1201)
y = sf.gaussian(x, 480, 25, area=12) + sf.lorentzian(x, 540, 12, area=8) + 0.05

fit = sf.fit_peaks(x, y, [("gaussian",   dict(x0=478, area=10, fwhm=20)),
                          ("lorentzian", dict(x0=542, area=10, fwhm=20))],
                   baseline_order=1)
for p in fit.peaks:
    print(p.shape, round(p.x0, 2), round(p.area, 3))

print(sf.voigt_fwhm(1.0, 1.0))      # 1.6376
print(sf.nm_to_wavenumber(500))     # 20000 cm^-1
```

## Notebook

[`notebooks/spectral_analysis.ipynb`](notebooks/spectral_analysis.ipynb) covers five topics:

- line shapes and the Voigt width rule
- a Beer–Lambert calibration curve
- ALS baseline removal
- overlapping-peak fitting, comparing two approaches against the known truth:
  - ALS correction followed by a fit, which comes out biased low
  - a joint fit of peaks and baseline, which recovers the true values
- unmixing a three-component spectrum

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- every line shape integrates to its area over the whole real line
- the Gaussian and Lorentzian FWHM
- the Voigt profile reduces to its Gaussian and Lorentzian limits
- the Olivero formula agrees with the measured Voigt width
- Beer–Lambert round trips and unit conversions
- a noisy three-peak fit recovers positions, areas and the baseline
- the ALS baseline stays within 2 % of the peak height
- all three peaks are detected
- exact unmixing, both non-negative and unconstrained
- invalid input is rejected

## Scope and limitations

- Fits are local least squares, so give reasonable starting guesses (`detect_peaks` helps).
- ALS is convenient but can take part of broad peak tails. When the background is smooth, fit the baseline together with the peaks; the notebook shows the difference.

## References

- P. H. C. Eilers and H. F. M. Boelens, *Baseline Correction with Asymmetric Least Squares Smoothing* (2005).
- J. J. Olivero and R. L. Longbothum, *J. Quant. Spectrosc. Radiat. Transfer* 17, 233 (1977).
- D. C. Harris, *Quantitative Chemical Analysis*, 10th ed., W. H. Freeman (2020).

## License

MIT
