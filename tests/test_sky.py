import numpy as np
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import AtmosphericExtinction, InstrumentSimulator, ThroughputCurve
from simulator.components import (
    DESI_SKY_BRIGHT,
    DESI_SKY_DARK,
    DESI_SKY_GREY,
)


def test_desi_sky_spectra_have_surface_brightness_units():
    for sky in (DESI_SKY_DARK, DESI_SKY_GREY, DESI_SKY_BRIGHT):
        wavelength, surface_brightness = sky.spectrum()

        assert wavelength.unit.is_equivalent(u.AA)
        assert surface_brightness.unit.is_equivalent(
            u.erg / u.s / u.cm**2 / u.AA / u.arcsec**2
        )
        assert wavelength.ndim == 1
        assert wavelength.shape == surface_brightness.shape
        assert np.all(np.diff(wavelength.to_value(u.AA)) > 0)


def test_desi_sky_brightness_ordering():
    values = []
    for sky in (DESI_SKY_DARK, DESI_SKY_GREY, DESI_SKY_BRIGHT):
        wavelength, surface_brightness = sky.spectrum()
        values.append(
            np.interp(
                5000.0,
                wavelength.to_value(u.AA),
                surface_brightness.to_value(
                    u.erg / u.s / u.cm**2 / u.AA / u.arcsec**2
                ),
            )
        )

    assert values[0] < values[1] < values[2]


def test_fiber_sky_area_uses_telescope_plate_scale(small_spectrograph):
    simulator = InstrumentSimulator(
        spectrograph=small_spectrograph,
        throughputs=[],
        sky=DESI_SKY_DARK,
    )

    assert_allclose(
        simulator.fiber_sky_area.to_value(u.arcsec**2),
        5.5804857895025055,
        rtol=1e-12,
    )


def test_sky_is_not_attenuated_by_atmosphere(small_spectrograph):
    one_airmass = InstrumentSimulator(
        spectrograph=small_spectrograph,
        atmosphere=AtmosphericExtinction(airmass=1.0),
        sky=DESI_SKY_DARK,
    )
    two_airmass = InstrumentSimulator(
        spectrograph=small_spectrograph,
        atmosphere=AtmosphericExtinction(airmass=2.0),
        sky=DESI_SKY_DARK,
    )

    first = one_airmass.render_sky_electrons(10 * u.s)
    second = two_airmass.render_sky_electrons(10 * u.s)

    assert_allclose(first.value, second.value, rtol=0, atol=0)


def test_sky_uses_downstream_instrument_throughput(small_spectrograph):
    unity = ThroughputCurve(
        np.array([3000.0, 11000.0]) * u.AA,
        np.ones(2),
    )
    half = ThroughputCurve(
        np.array([3000.0, 11000.0]) * u.AA,
        np.full(2, 0.5),
    )
    full_simulator = InstrumentSimulator(
        spectrograph=small_spectrograph,
        throughputs=[unity],
        sky=DESI_SKY_DARK,
    )
    half_simulator = InstrumentSimulator(
        spectrograph=small_spectrograph,
        throughputs=[half],
        sky=DESI_SKY_DARK,
    )

    full = full_simulator.render_sky_electrons(10 * u.s)
    attenuated = half_simulator.render_sky_electrons(10 * u.s)

    assert_allclose(
        attenuated.sum().to_value(u.electron),
        0.5 * full.sum().to_value(u.electron),
        rtol=1e-12,
    )


def test_configured_sky_is_added_to_source_image(small_spectrograph):
    unity = ThroughputCurve(
        np.array([3000.0, 11000.0]) * u.AA,
        np.ones(2),
    )
    simulator = InstrumentSimulator(
        spectrograph=small_spectrograph,
        throughputs=[unity],
        sky=DESI_SKY_DARK,
    )
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    zero_flux = np.zeros(2) * u.erg / u.s / u.cm**2 / u.AA

    image = simulator.render_electrons(wavelength, zero_flux, 10 * u.s)

    assert image.sum().to_value(u.electron) > 0
