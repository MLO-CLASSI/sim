import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose
from specreduce.wavesol1d import WavelengthSolution1D

from simulator import DetectorModel, InstrumentSimulator, SpectrographModel


def test_imx571_reference_geometry(imx571_spectrograph):
    model = imx571_spectrograph

    assert_allclose(model.central_wavelength.to_value(u.AA), 6263.304030251207, rtol=1e-12)
    assert_allclose(model.dispersion.to_value(u.AA / u.pixel), 1.1777480847183388, rtol=1e-12)
    assert_allclose(model.magnification.value, 0.5555555555555556, rtol=1e-12)
    assert_allclose(model.anamorphic_factor.value, 0.9024739339201835, rtol=1e-12)
    assert_allclose(model.fiber_pitch_px.to_value(u.pixel), 36.938534278959814, rtol=1e-12)
    assert_allclose(model.spatial_fwhm_px.to_value(u.pixel), 15.514184397163122, rtol=1e-12)
    assert_allclose(model.spectral_fwhm_px.to_value(u.pixel), 14.001147024470933, rtol=1e-12)


def test_spectrograph_exposes_specreduce_wavelength_solution(imx571_spectrograph):
    solution = imx571_spectrograph.wavelength_solution

    assert isinstance(solution, WavelengthSolution1D)
    assert solution.bounds_pix == (0, imx571_spectrograph.detector.nx)
    assert solution.unit == u.AA


def test_wavelength_solution_matches_grating_geometry(imx571_spectrograph):
    model = imx571_spectrograph
    pixels = np.linspace(0, model.detector.nx - 1, 257)
    center = model.x_center.to_value(u.pixel)
    detector_offset = center - pixels
    field_angle = np.arctan(
        detector_offset
        * (model.detector.pixel_size / model.camera_focal_length).to_value(
            u.dimensionless_unscaled
        )
    ) * u.rad
    diffraction_angle = model.diffraction_angle + field_angle
    expected = (
        model.groove_spacing
        * (np.sin(model.incidence_angle) + np.sin(diffraction_angle))
        / model.diffraction_order
    ).to_value(u.AA)

    actual = model.wavelength_solution.pix_to_wav(pixels)

    assert_allclose(actual, expected, rtol=0, atol=2e-8)


def test_simulator_exposes_binned_wavelength_solution(imx571_spectrograph):
    simulator = InstrumentSimulator(
        imx571_spectrograph,
        binning=2,
        throughputs=[],
    )

    assert simulator.wavelength_solution is simulator.readout_spectrograph.wavelength_solution
    assert simulator.wavelength_solution.bounds_pix == (0, 3122)


def test_central_wavelength_maps_to_detector_center(imx571_spectrograph):
    model = imx571_spectrograph
    x = model.wavelength_to_x(model.central_wavelength)
    assert_allclose(x.to_value(u.pixel), model.x_center.to_value(u.pixel), atol=1e-5)


def test_wavelength_x_round_trip(imx571_spectrograph):
    model = imx571_spectrograph
    wavelength = np.array([4000.0, 5000.0, 6263.304030251207, 7500.0, 9000.0]) * u.AA

    recovered = model.x_to_wavelength(model.wavelength_to_x(wavelength))

    assert_allclose(recovered.to_value(u.AA), wavelength.to_value(u.AA), rtol=0, atol=1e-4)


def test_legacy_mapping_methods_delegate_to_wavelength_solution(imx571_spectrograph):
    model = imx571_spectrograph
    pixels = np.array([500.25, 1500.5, 3000.75])

    assert_allclose(
        model.x_to_wavelength(pixels * u.pixel).to_value(u.AA),
        model.wavelength_solution.pix_to_wav(pixels),
        rtol=0,
        atol=0,
    )


def test_wavelength_direction_flag_changes_only_detector_orientation(imx571_detector):
    kwargs = dict(
        detector=imx571_detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
    )
    decreasing = SpectrographModel(**kwargs, wavelength_increases_with_x=False)
    increasing = SpectrographModel(**kwargs, wavelength_increases_with_x=True)
    wavelength = 7000 * u.AA

    dx_decreasing = decreasing.wavelength_to_x(wavelength) - decreasing.x_center
    dx_increasing = increasing.wavelength_to_x(wavelength) - increasing.x_center

    assert_allclose(dx_increasing.to_value(u.pixel), -dx_decreasing.to_value(u.pixel), atol=1e-5)
    assert_allclose(
        increasing.x_to_wavelength(increasing.wavelength_to_x(wavelength)).to_value(u.AA),
        wavelength.to_value(u.AA),
        atol=1e-4,
    )


def test_fiber_trace_centers_are_symmetric_and_evenly_spaced(imx571_spectrograph):
    model = imx571_spectrograph
    centers = model.fiber_trace_centers().to_value(u.pixel)

    assert len(centers) == 7
    assert_allclose(centers[3], model.trace_y.to_value(u.pixel), atol=1e-12)
    assert_allclose(np.diff(centers), model.fiber_pitch_px.to_value(u.pixel), atol=1e-12)
    assert_allclose(centers - model.trace_y.to_value(u.pixel), -(centers[::-1] - model.trace_y.to_value(u.pixel)), atol=1e-12)


def test_unreachable_wavelength_is_rejected(imx571_spectrograph):
    with pytest.raises(ValueError, match="not physically reachable"):
        imx571_spectrograph.wavelength_to_x(1.0e8 * u.AA)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diffraction_order": 0}, "diffraction_order"),
        ({"fiber_count": 0}, "fiber_count"),
        ({"render_sampling_px": 0}, "render_sampling_px"),
    ],
)
def test_invalid_spectrograph_configuration_is_rejected(imx571_detector, kwargs, message):
    config = dict(
        detector=imx571_detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
    )
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SpectrographModel(**config)
