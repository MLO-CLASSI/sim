import numpy as np
from astropy import units as u
from numpy.testing import assert_allclose

from simulator import DetectorModel


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
