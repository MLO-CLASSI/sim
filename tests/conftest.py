import numpy as np
import pytest
from astropy import units as u

from simulator import DetectorModel, InstrumentSimulator, SpectrographModel, ThroughputCurve


@pytest.fixture
def imx571_detector():
    return DetectorModel(
        nx=6244,
        ny=4168,
        pixel_size=3.76 * u.um,
    )


@pytest.fixture
def imx571_spectrograph(imx571_detector):
    return SpectrographModel(
        detector=imx571_detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        diffraction_order=1,
        fiber_count=7,
        fiber_pitch=250 * u.um,
        render_sampling_px=0.5,
    )


@pytest.fixture
def small_spectrograph():
    detector = DetectorModel(
        nx=256,
        ny=192,
        pixel_size=3.76 * u.um,
    )
    return SpectrographModel(
        detector=detector,
        groove_density=300 / u.mm,
        incidence_angle=32 * u.deg,
        diffraction_angle=-20 * u.deg,
        collimator_focal_length=180 * u.mm,
        camera_focal_length=100 * u.mm,
        fiber_core_diameter=105 * u.um,
        diffraction_order=1,
        fiber_count=1,
        fiber_pitch=250 * u.um,
        render_sampling_px=0.5,
    )


@pytest.fixture
def unity_simulator(small_spectrograph):
    throughput = ThroughputCurve(
        wavelength=np.array([3000.0, 10000.0]) * u.AA,
        throughput=np.ones(2),
        name="unity",
    )
    return InstrumentSimulator(
        spectrograph=small_spectrograph,
        throughputs=[throughput],
    )
