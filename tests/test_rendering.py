import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import DetectorModel, InstrumentSimulator, SpectrographModel, ThroughputCurve
from simulator.core import FLUX_DENSITY_UNIT, TELESCOPE_AREA, f_lambda_to_photon_flux_density


def test_trapezoid_bin_widths_preserve_total_span():
    wavelength = np.array([4000.0, 4010.0, 4040.0, 4100.0]) * u.AA
    widths = InstrumentSimulator._trapezoid_bin_widths(wavelength)

    assert_allclose(widths.to_value(u.AA), [5.0, 20.0, 45.0, 30.0])
    assert_allclose(widths.sum().to_value(u.AA), 100.0)


def test_resampling_preserves_piecewise_linear_flux_integral(unity_simulator):
    wavelength = np.array([6100.0, 6250.0, 6400.0]) * u.AA
    flux = np.array([[1.0, 3.0, 2.0]]) * FLUX_DENSITY_UNIT

    render_wavelength, render_flux = unity_simulator._resample_for_detector(wavelength, flux)

    before = np.trapezoid(flux[0].value, x=wavelength.to_value(u.AA))
    after = np.trapezoid(render_flux[0].value, x=render_wavelength.to_value(u.AA))
    assert_allclose(after, before, rtol=1e-13)


def test_resampling_limits_spacing_in_detector_coordinates(unity_simulator):
    wavelength = np.array([6100.0, 6400.0]) * u.AA

    render_wavelength, _ = unity_simulator._resample_for_detector(
        wavelength,
        np.ones((1, 2)) * FLUX_DENSITY_UNIT,
    )
    x = unity_simulator.spectrograph.wavelength_to_x(render_wavelength).to_value(u.pixel)

    assert np.max(np.abs(np.diff(x))) <= unity_simulator.spectrograph.render_sampling_px * (1 + 1e-12)


def test_resampling_rejects_duplicate_wavelengths(unity_simulator):
    wavelength = np.array([6200.0, 6200.0]) * u.AA
    flux = np.ones((1, 2)) * FLUX_DENSITY_UNIT

    with pytest.raises(ValueError, match="unique"):
        unity_simulator._resample_for_detector(wavelength, flux)


def test_gaussian_packet_deposition_conserves_counts_away_from_edges():
    image = np.zeros((101, 101)) * u.electron

    InstrumentSimulator._deposit_gaussian_packets(
        image=image,
        x_centers=np.array([50.25]) * u.pixel,
        y_centers=np.array([49.75]) * u.pixel,
        counts=np.array([1234.5]) * u.electron,
        sigma_x=2.0 * u.pixel,
        sigma_y=3.0 * u.pixel,
        radius_sigma=4.0,
    )

    assert_allclose(image.sum().to_value(u.electron), 1234.5, rtol=1e-14)


def test_gaussian_packet_deposition_loses_off_detector_light():
    image = np.zeros((41, 41)) * u.electron

    InstrumentSimulator._deposit_gaussian_packets(
        image=image,
        x_centers=np.array([0.0]) * u.pixel,
        y_centers=np.array([20.0]) * u.pixel,
        counts=np.array([1000.0]) * u.electron,
        sigma_x=2.0 * u.pixel,
        sigma_y=2.0 * u.pixel,
        radius_sigma=4.0,
    )

    assert 0 < image.sum().to_value(u.electron) < 1000.0


def test_rendered_total_matches_integrated_photon_count(small_spectrograph):
    small_spectrograph.render_sampling_px = 100.0
    throughput_value = 0.4
    throughput = ThroughputCurve(
        np.array([6000.0, 6500.0]) * u.AA,
        np.array([throughput_value, throughput_value]),
    )
    simulator = InstrumentSimulator(small_spectrograph, [throughput])
    wavelength = small_spectrograph.x_to_wavelength(np.array([120.0, 127.5, 135.0]) * u.pixel)
    flux = np.array([1.0, 2.0, 1.5]) * 1.0e-15 * FLUX_DENSITY_UNIT
    exposure = 30 * u.s

    image = simulator.render_electrons(wavelength, flux, exposure)

    widths = simulator._trapezoid_bin_widths(wavelength)
    photon_rate_density = f_lambda_to_photon_flux_density(wavelength, flux, TELESCOPE_AREA)
    expected = np.sum(
        photon_rate_density
        * widths
        * exposure
        * throughput_value
    ).to_value(u.dimensionless_unscaled)
    assert_allclose(image.sum().to_value(u.electron), expected, rtol=1e-12)



def test_rendering_is_invariant_to_input_wavelength_order(unity_simulator):
    wavelength = np.array([6220.0, 6200.0, 6210.0]) * u.AA
    flux = np.array([3.0, 1.0, 2.0]) * 1.0e-15 * FLUX_DENSITY_UNIT

    unsorted = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)
    order = np.argsort(wavelength.value)
    sorted_image = unity_simulator.render_electrons(wavelength[order], flux[order], 10 * u.s)

    assert_allclose(unsorted.value, sorted_image.value, rtol=0, atol=0)


def test_rendered_counts_scale_linearly_with_exposure(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    short = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)
    long = unity_simulator.render_electrons(wavelength, flux, 40 * u.s)

    assert_allclose(long.value, 4.0 * short.value, rtol=1e-13, atol=0)


def test_scalar_fiber_coupling_scales_source_counts(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    full = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)
    quarter = unity_simulator.render_electrons(
        wavelength,
        flux,
        10 * u.s,
        fiber_coupling_efficiency=0.25,
    )

    assert_allclose(quarter.sum().value, 0.25 * full.sum().value, rtol=1e-12)


def test_per_fiber_coupling_scales_each_trace_independently():
    detector = DetectorModel(nx=256, ny=256, pixel_size=3.76 * u.um)
    spectrograph = SpectrographModel(
        detector=detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        fiber_count=3,
        fiber_pitch=500 * u.um,
        render_sampling_px=100.0,
    )
    simulator = InstrumentSimulator(
        spectrograph,
        [ThroughputCurve(np.array([6000.0, 6500.0]) * u.AA, np.ones(2))],
    )
    wavelength = spectrograph.x_to_wavelength(np.array([124.0, 131.0]) * u.pixel)
    flux = np.ones((3, 2)) * 1.0e-15 * FLUX_DENSITY_UNIT
    coupling = np.array([0.0, 0.5, 1.0])

    image = simulator.render_electrons(
        wavelength,
        flux,
        10 * u.s,
        fiber_coupling_efficiency=coupling,
    )
    centers = np.rint(spectrograph.fiber_trace_centers().to_value(u.pixel)).astype(int)
    half_spacing = int(np.floor(0.5 * spectrograph.fiber_pitch_px.to_value(u.pixel)))
    trace_sums = []
    for center in centers:
        trace_sums.append(
            image[center - half_spacing:center + half_spacing + 1].sum().to_value(u.electron)
        )

    assert_allclose(trace_sums[0], 0.0, atol=1e-12)
    assert_allclose(trace_sums[1] / trace_sums[2], 0.5, rtol=1e-12)



def test_invalid_fiber_coupling_vector_shape_is_rejected():
    detector = DetectorModel(nx=64, ny=64, pixel_size=3.76 * u.um)
    spectrograph = SpectrographModel(
        detector=detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        fiber_count=3,
        fiber_pitch=250 * u.um,
    )
    simulator = InstrumentSimulator(spectrograph, [])
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones((3, 2)) * 1.0e-15 * FLUX_DENSITY_UNIT

    with pytest.raises(ValueError, match="one value per fiber"):
        simulator.render_electrons(
            wavelength,
            flux,
            1 * u.s,
            fiber_coupling_efficiency=np.array([0.5, 0.5]),
        )


@pytest.mark.parametrize("coupling", [-0.01, 1.01, np.nan])
def test_invalid_fiber_coupling_is_rejected(unity_simulator, coupling):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    with pytest.raises(ValueError, match="between 0 and 1"):
        unity_simulator.render_electrons(
            wavelength,
            flux,
            1 * u.s,
            fiber_coupling_efficiency=coupling,
        )


def test_vignetting_multiplies_rendered_image(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    unvignetted = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)
    factor = np.full(unvignetted.shape, 0.25)
    vignetted = unity_simulator.render_electrons(wavelength, flux, 10 * u.s, vignetting=factor)

    assert_allclose(vignetted.value, 0.25 * unvignetted.value, rtol=1e-14, atol=0)



def test_vignetting_rejects_wrong_shape(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    with pytest.raises(ValueError, match="same shape"):
        unity_simulator.render_electrons(
            wavelength,
            flux,
            1 * u.s,
            vignetting=np.ones((2, 2)),
        )


def test_simulate_without_noise_matches_render_electrons(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT

    rendered = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)
    simulated = unity_simulator.simulate(wavelength, flux, 10 * u.s, add_noise=False)

    assert simulated.unit == u.electron
    assert_allclose(simulated.value, rendered.value, rtol=0, atol=0)


def test_negative_input_flux_does_not_generate_negative_electrons(unity_simulator):
    wavelength = np.array([6200.0, 6210.0]) * u.AA
    flux = np.array([-1.0, -1.0]) * 1.0e-15 * FLUX_DENSITY_UNIT

    image = unity_simulator.render_electrons(wavelength, flux, 10 * u.s)

    assert np.all(image.value == 0)


def test_multifiber_simulation_requires_one_spectrum_per_fiber():
    detector = DetectorModel(nx=64, ny=64, pixel_size=3.76 * u.um)
    spectrograph = SpectrographModel(
        detector=detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        fiber_count=3,
        fiber_pitch=250 * u.um,
    )
    simulator = InstrumentSimulator(spectrograph, [])

    with pytest.raises(ValueError, match=r"shape \(fiber_count, n_wavelength\)"):
        simulator.render_electrons(
            np.array([6200.0, 6210.0]) * u.AA,
            np.ones(2) * 1.0e-15 * FLUX_DENSITY_UNIT,
            1 * u.s,
        )
