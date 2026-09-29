import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import AtmosphericExtinction, InstrumentSimulator, ThroughputCurve, f_lambda_to_photon_flux_density
from simulator.components import CLAUD_50INCH


def test_throughput_curve_interpolates_across_compatible_units():
    curve = ThroughputCurve(
        wavelength=np.array([400.0, 500.0, 600.0]) * u.nm,
        throughput=np.array([0.2, 0.4, 0.8]),
    )

    result = curve(np.array([3500.0, 4500.0, 5500.0, 6500.0]) * u.AA)

    assert_allclose(result, [0.0, 0.3, 0.6, 0.0], atol=1e-15)


def test_throughput_curve_rejects_dimensionless_wavelengths():
    with pytest.raises(u.UnitConversionError):
        ThroughputCurve(
            wavelength=np.array([400.0, 500.0]),
            throughput=np.array([0.5, 0.5]),
        )


def test_throughput_curve_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ThroughputCurve(
            wavelength=np.array([400.0, 500.0]) * u.nm,
            throughput=np.array([0.5]),
        )



def test_throughput_curve_from_csv_uses_nm_by_default(tmp_path):
    path = tmp_path / "throughput.csv"
    path.write_text("400,0.2\n500,0.6\n")

    curve = ThroughputCurve.from_csv(path)

    assert_allclose(curve.wavelength.to_value(u.nm), [400.0, 500.0])
    assert_allclose(curve(np.array([4000.0, 5000.0]) * u.AA), [0.2, 0.6])


def test_combined_throughput_multiplies_components(small_spectrograph):
    wave = np.array([4000.0, 5000.0, 6000.0]) * u.AA
    first = ThroughputCurve(wave, np.array([0.8, 0.7, 0.6]))
    second = ThroughputCurve(wave, np.array([0.5, 0.4, 0.3]))
    simulator = InstrumentSimulator(small_spectrograph, [first, second])

    assert_allclose(simulator.combined_throughput(wave), [0.4, 0.28, 0.18], atol=1e-15)



def test_telescope_collecting_area_includes_central_obstruction():
    assert_allclose(CLAUD_50INCH.collecting_area.to_value(u.m**2), 1.1167380135807468, rtol=1e-14)


def test_atmospheric_extinction_decreases_with_airmass():
    wavelength = np.array([450.0, 600.0, 800.0]) * u.nm
    one_airmass = AtmosphericExtinction(airmass=1.0)(wavelength)
    two_airmass = AtmosphericExtinction(airmass=2.0)(wavelength)

    assert np.all(two_airmass < one_airmass)
    assert np.all((one_airmass > 0) & (one_airmass <= 1))


def test_f_lambda_to_photon_flux_density_reference_value():
    rate = f_lambda_to_photon_flux_density(
        5000 * u.AA,
        1.0e-15 * u.erg / (u.s * u.cm**2 * u.AA),
        100 * u.cm**2,
    )

    assert rate.unit.is_equivalent(1 / (u.s * u.AA))
    assert_allclose(rate.to_value(1 / (u.s * u.AA)), 0.025170582837713548, rtol=1e-12)


def test_photon_flux_density_scales_linearly_with_flux_and_area():
    wavelength = 6500 * u.AA
    flux = 2.0e-16 * u.erg / (u.s * u.cm**2 * u.AA)

    base = f_lambda_to_photon_flux_density(wavelength, flux, 50 * u.cm**2)
    scaled = f_lambda_to_photon_flux_density(wavelength, 3 * flux, 2 * 50 * u.cm**2)

    assert_allclose(scaled.value / base.value, 6.0, atol=1e-12)
