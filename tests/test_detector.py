import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import DetectorModel, DetectorReadout


class DeterministicRNG:
    def __init__(self, read_offset=0.0):
        self.read_offset = read_offset

    def poisson(self, expected):
        return np.asarray(expected, dtype=float)

    def normal(self, loc, scale, size):
        return np.full(size, self.read_offset, dtype=float)


def test_detector_applies_dark_gain_and_bias():
    detector = DetectorModel(
        nx=2,
        ny=1,
        pixel_size=3.76 * u.um,
        gain=2.0 * u.electron / u.adu,
        read_noise=0.0 * u.electron,
        dark_current=3.0 * u.electron / u.s,
        bias=100.0 * u.adu,
    )
    image = np.array([[10.0, 20.0]]) * u.electron

    result = detector.apply_noise(image, 2 * u.s, DeterministicRNG())

    assert_allclose(result.to_value(u.adu), [[108.0, 113.0]])


def test_full_well_saturates_charge_before_read_noise():
    detector = DetectorModel(
        nx=1,
        ny=1,
        pixel_size=3.76 * u.um,
        gain=1.0 * u.electron / u.adu,
        read_noise=2.0 * u.electron,
        dark_current=0.0 * u.electron / u.s,
        bias=0.0 * u.adu,
        full_well=25.0 * u.electron,
    )

    result = detector.apply_noise(
        np.array([[30.0]]) * u.electron,
        1 * u.s,
        DeterministicRNG(read_offset=5.0),
    )

    assert_allclose(result.to_value(u.adu), [[30.0]])


def test_detector_noise_is_reproducible_for_fixed_seed():
    detector = DetectorModel(
        nx=8,
        ny=8,
        pixel_size=3.76 * u.um,
        gain=1.3 * u.electron / u.adu,
        read_noise=2.0 * u.electron,
        dark_current=0.2 * u.electron / u.s,
        bias=100.0 * u.adu,
    )
    image = np.full((8, 8), 20.0) * u.electron

    first = detector.apply_noise(image, 30 * u.s, np.random.default_rng(12345))
    second = detector.apply_noise(image, 30 * u.s, np.random.default_rng(12345))

    assert_allclose(first.value, second.value, rtol=0, atol=0)


def test_readout_scales_output_geometry_and_effective_noise():
    detector = DetectorModel(
        nx=6244,
        ny=4168,
        pixel_size=3.76 * u.um,
        gain=0.5 * u.electron / u.adu,
        read_noise=1.0 * u.electron,
        dark_current=0.002 * u.electron / u.s,
        bias=200.0 * u.adu,
        full_well=50000.0 * u.electron,
    )
    readout = DetectorReadout(detector, binning=2)

    assert detector.nx == 6244
    assert detector.ny == 4168
    assert_allclose(detector.pixel_size.to_value(u.um), 3.76)
    assert_allclose(detector.read_noise.to_value(u.electron), 1.0)
    assert_allclose(detector.dark_current.to_value(u.electron / u.s), 0.002)
    assert_allclose(detector.full_well.to_value(u.electron), 50000.0)

    assert readout.nx == 3122
    assert readout.ny == 2084
    assert_allclose(readout.pixel_size.to_value(u.um), 7.52)
    assert_allclose(readout.read_noise.to_value(u.electron), 2.0)
    assert_allclose(readout.dark_current.to_value(u.electron / u.s), 0.008)
    assert_allclose(readout.gain.to_value(u.electron / u.adu), 0.5)
    assert_allclose(readout.bias.to_value(u.adu), 200.0)


def test_readout_sums_native_pixels_before_gain_and_bias():
    detector = DetectorModel(
        nx=2,
        ny=2,
        pixel_size=3.76 * u.um,
        gain=2.0 * u.electron / u.adu,
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.s,
        bias=100.0 * u.adu,
    )
    readout = DetectorReadout(detector, binning=2)
    image = np.array([[2.0, 4.0], [6.0, 8.0]]) * u.electron

    result = readout.apply_noise(image, 1 * u.s, DeterministicRNG())

    assert_allclose(result.to_value(u.adu), [[110.0]])


def test_native_pixel_saturation_occurs_before_binning():
    detector = DetectorModel(
        nx=2,
        ny=2,
        pixel_size=3.76 * u.um,
        gain=1.0 * u.electron / u.adu,
        read_noise=0.0 * u.electron,
        dark_current=0.0 * u.electron / u.s,
        bias=0.0 * u.adu,
        full_well=25.0 * u.electron,
    )
    readout = DetectorReadout(detector, binning=2)
    image = np.array([[30.0, 0.0], [0.0, 0.0]]) * u.electron

    result = readout.apply_noise(image, 1 * u.s, DeterministicRNG())

    assert_allclose(result.to_value(u.adu), [[25.0]])


@pytest.mark.parametrize("binning", [0, -1, 1.5, True])
def test_readout_rejects_invalid_binning(binning):
    detector = DetectorModel(nx=8, ny=8, pixel_size=3.76 * u.um)

    with pytest.raises(ValueError, match="positive integer"):
        DetectorReadout(detector, binning=binning)


def test_readout_rejects_binning_that_does_not_divide_dimensions():
    detector = DetectorModel(nx=5, ny=4, pixel_size=3.76 * u.um)

    with pytest.raises(ValueError, match="evenly divide"):
        DetectorReadout(detector, binning=2)
