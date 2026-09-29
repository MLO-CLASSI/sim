import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import AtmosphericExtinction, InstrumentSimulator, SpectrographModel
from simulator.components import (
    CANON_EF100_F2,
    CLASSI_FIBER,
    CLAUD_50INCH,
    CLAUD_50INCH,
    E02_PICKOFF,
    FGL400S,
    FLI_AR571,
    NEWPORT_MASTER_1229,
    NEWPORT_MASTER_1294,
    THORLABS_AC508_180_AB,
    FiberModel,
)


def test_claud_telescope_geometry():
    assert_allclose(
        CLAUD_50INCH.collecting_area.to_value(u.m**2),
        1.1167380135807468,
        rtol=1e-14,
    )
    assert_allclose(
        CLAUD_50INCH.plate_scale.to_value(u.arcsec / u.mm),
        25.386438307334937,
        rtol=1e-12,
    )


def test_fiber_throughput_depends_on_length():
    wavelength = np.array([400.0, 500.0, 700.0]) * u.nm
    zero_length = CLASSI_FIBER.throughput_curve(0 * u.m)(wavelength)
    ten_meters = CLASSI_FIBER.throughput_curve(10 * u.m)(wavelength)

    assert_allclose(zero_length, np.ones(wavelength.size), atol=1e-14)
    assert np.all(ten_meters <= zero_length)
    assert np.any(ten_meters < zero_length)


def test_fiber_rejects_negative_length():
    with pytest.raises(ValueError, match="cannot be negative"):
        CLASSI_FIBER.throughput_curve(-1 * u.m)


def test_grating_component_carries_geometry_and_efficiency():
    assert_allclose(
        NEWPORT_MASTER_1294.groove_density.to_value(1 / u.mm),
        300.0,
    )
    throughput = NEWPORT_MASTER_1294.throughput_curve()(600 * u.nm)
    assert 0 < throughput < 1


def test_polarized_grating_curves_are_combined():
    throughput = NEWPORT_MASTER_1229.throughput_curve()
    values = throughput(np.array([400.0, 500.0, 700.0]) * u.nm)

    assert np.all(np.isfinite(values))
    assert np.all((values >= 0) & (values <= 1))


def test_named_optics_load_their_throughput_data():
    assert_allclose(
        THORLABS_AC508_180_AB.focal_length.to_value(u.mm),
        180.0,
    )
    assert_allclose(CANON_EF100_F2.focal_length.to_value(u.mm), 100.0)

    wavelength = 600 * u.nm
    assert 0 < THORLABS_AC508_180_AB.throughput_curve()(wavelength) <= 1
    assert 0 < E02_PICKOFF.throughput_curve()(wavelength) <= 1


def test_camera_component_owns_qe_and_window_curves():
    wavelength = 600 * u.nm

    assert 0 < FLI_AR571.qe_curve()(wavelength) <= 1
    assert 0 < FLI_AR571.window_curve()(wavelength) <= 1


def test_spectrograph_can_be_constructed_from_components():
    model = SpectrographModel.from_components(
        detector=FLI_AR571,
        grating=NEWPORT_MASTER_1294,
        collimator=THORLABS_AC508_180_AB,
        camera_lens=CANON_EF100_F2,
        fiber=CLASSI_FIBER,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        fiber_count=7,
        fiber_pitch=250 * u.um,
        optical_elements=(E02_PICKOFF, FGL400S),
    )

    assert model.detector is FLI_AR571
    assert model.grating is NEWPORT_MASTER_1294
    assert model.collimator is THORLABS_AC508_180_AB
    assert model.camera_lens is CANON_EF100_F2
    assert model.fiber is CLASSI_FIBER
    assert model.optical_elements == (E02_PICKOFF, FGL400S)
    assert_allclose(model.groove_density.to_value(1 / u.mm), 300.0)
    assert_allclose(model.collimator_focal_length.to_value(u.mm), 180.0)
    assert_allclose(model.camera_focal_length.to_value(u.mm), 100.0)
    assert_allclose(model.fiber_core_diameter.to_value(u.um), 105.0)


def test_spectrograph_component_constructor_requires_fiber_core_size():
    fiber = FiberModel(
        name="test fiber",
        attenuation_resource="uvns_attenuation",
    )

    with pytest.raises(ValueError, match="core_diameter"):
        SpectrographModel.from_components(
            detector=FLI_AR571,
            grating=NEWPORT_MASTER_1294,
            collimator=THORLABS_AC508_180_AB,
            camera_lens=CANON_EF100_F2,
            fiber=fiber,
            incidence_angle=32 * u.deg,
            diffraction_angle=-20 * u.deg,
        )


def test_simulator_builds_throughput_chain_from_components():
    spectrograph = SpectrographModel.from_components(
        detector=FLI_AR571,
        grating=NEWPORT_MASTER_1294,
        collimator=THORLABS_AC508_180_AB,
        camera_lens=CANON_EF100_F2,
        fiber=CLASSI_FIBER,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        optical_elements=(E02_PICKOFF, FGL400S),
    )
    atmosphere = AtmosphericExtinction(airmass=1.3)

    simulator = InstrumentSimulator(
        spectrograph=spectrograph,
        telescope=CLAUD_50INCH,
        atmosphere=atmosphere,
    )

    names = [curve.name for curve in simulator.throughputs]
    assert names == [
        "atmosphere",
        E02_PICKOFF.name,
        FGL400S.name,
        CLASSI_FIBER.name,
        THORLABS_AC508_180_AB.name,
        NEWPORT_MASTER_1294.name,
        CANON_EF100_F2.name,
        f"{FLI_AR571.name} window",
        f"{FLI_AR571.name} QE",
    ]


def test_claud_telescope_aliases_are_identical():
    assert CLAUD_50INCH is CLAUD_50INCH


def test_simulator_defaults_to_claud_telescope(small_spectrograph):
    from simulator import InstrumentSimulator

    simulator = InstrumentSimulator(small_spectrograph, throughputs=[])

    assert simulator.telescope is CLAUD_50INCH
