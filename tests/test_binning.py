import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import DetectorModel, InstrumentSimulator, SpectrographModel, ThroughputCurve
from simulator.core import FLUX_DENSITY_UNIT


def make_spectrograph():
    detector = DetectorModel(
        nx=64,
        ny=48,
        pixel_size=3.76 * u.um,
        gain=1.0 * u.electron / u.adu,
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.s,
        bias=0.0 * u.adu,
    )
    return SpectrographModel(
        detector=detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        render_sampling_px=0.5,
    )


def make_simulator(spectrograph, binning):
    throughput = ThroughputCurve(
        np.array([3000.0, 10000.0]) * u.AA,
        np.ones(2),
    )
    return InstrumentSimulator(
        spectrograph=spectrograph,
        binning=binning,
        throughputs=[throughput],
    )


def test_simulator_binning_does_not_mutate_detector():
    spectrograph = make_spectrograph()
    detector = spectrograph.detector

    sim_2x2 = make_simulator(spectrograph, 2)
    sim_4x4 = make_simulator(spectrograph, 4)

    assert detector.nx == 64
    assert detector.ny == 48
    assert_allclose(detector.pixel_size.to_value(u.um), 3.76)
    assert sim_2x2.readout.nx == 32
    assert sim_2x2.readout.ny == 24
    assert sim_4x4.readout.nx == 16
    assert sim_4x4.readout.ny == 12


def test_readout_spectrograph_uses_binned_pixel_coordinates():
    spectrograph = make_spectrograph()
    simulator = make_simulator(spectrograph, 2)
    readout_model = simulator.readout_spectrograph

    assert_allclose(
        readout_model.dispersion.to_value(u.AA / u.pixel),
        2 * spectrograph.dispersion.to_value(u.AA / u.pixel),
        rtol=1e-14,
    )
    assert_allclose(
        readout_model.fiber_pitch_px.to_value(u.pixel),
        0.5 * spectrograph.fiber_pitch_px.to_value(u.pixel),
        rtol=1e-14,
    )
    wavelength = np.array([5000.0, 6263.304030251207, 7500.0]) * u.AA
    recovered = readout_model.x_to_wavelength(
        readout_model.wavelength_to_x(wavelength)
    )
    assert_allclose(recovered.to_value(u.AA), wavelength.to_value(u.AA), atol=1e-9)


def test_binning_changes_output_shape_without_changing_total_expected_counts():
    spectrograph = make_spectrograph()
    native = make_simulator(spectrograph, 1)
    binned = make_simulator(spectrograph, 2)
    wavelength = spectrograph.x_to_wavelength(
        np.array([28.0, 31.5, 35.0]) * u.pixel
    )
    flux = np.array([1.0, 2.0, 1.0]) * 1.0e-15 * FLUX_DENSITY_UNIT

    native_image = native.render_electrons(wavelength, flux, 10 * u.s)
    binned_image = binned.render_electrons(wavelength, flux, 10 * u.s)

    assert native_image.shape == (48, 64)
    assert binned_image.shape == (24, 32)
    assert_allclose(
        binned_image.sum().to_value(u.electron),
        native_image.sum().to_value(u.electron),
        rtol=1e-14,
    )


def test_simulate_without_noise_returns_binned_electrons():
    spectrograph = make_spectrograph()
    simulator = make_simulator(spectrograph, 4)
    wavelength = spectrograph.x_to_wavelength(
        np.array([28.0, 31.5, 35.0]) * u.pixel
    )
    flux = np.ones(3) * 1.0e-15 * FLUX_DENSITY_UNIT

    rendered = simulator.render_electrons(wavelength, flux, 10 * u.s)
    simulated = simulator.simulate(
        wavelength,
        flux,
        10 * u.s,
        add_noise=False,
    )

    assert simulated.shape == (12, 16)
    assert simulated.unit == u.electron
    assert_allclose(simulated.value, rendered.value, rtol=0, atol=0)


def test_simulator_rejects_invalid_binning():
    spectrograph = make_spectrograph()

    with pytest.raises(ValueError, match="evenly divide"):
        make_simulator(spectrograph, 5)
