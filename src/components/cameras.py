
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from shared_data import CSV_FILES

from .optics import ThroughputCurve


@dataclass
class DetectorModel:
    nx: int
    ny: int
    pixel_size: u.Quantity
    gain: u.Quantity = 1.0 * u.electron / u.adu
    read_noise: u.Quantity = 0.0 * u.electron
    dark_current: u.Quantity = 0.0 * u.electron / u.s
    bias: u.Quantity = 0.0 * u.adu
    full_well: u.Quantity | None = None
    name: str = ""
    qe_resource: str | None = None
    window_resource: str | None = None

    def __post_init__(self) -> None:
        self.pixel_size = u.Quantity(self.pixel_size).to(u.um)
        self.gain = u.Quantity(self.gain).to(u.electron / u.adu)
        self.read_noise = u.Quantity(self.read_noise).to(u.electron)
        self.dark_current = u.Quantity(self.dark_current).to(u.electron / u.s)
        self.bias = u.Quantity(self.bias).to(u.adu)
        if self.full_well is not None:
            self.full_well = u.Quantity(self.full_well).to(u.electron)

    def qe_curve(self) -> ThroughputCurve:
        if self.qe_resource is None:
            raise ValueError(
                f"No QE resource is configured for {self.name or 'this detector'}."
            )
        return ThroughputCurve.from_csv(
            CSV_FILES[self.qe_resource],
            name=f"{self.name} QE" if self.name else "detector QE",
        )

    def window_curve(self) -> ThroughputCurve:
        if self.window_resource is None:
            raise ValueError(
                f"No window resource is configured for {self.name or 'this detector'}."
            )
        return ThroughputCurve.from_csv(
            CSV_FILES[self.window_resource],
            name=f"{self.name} window" if self.name else "detector window",
        )

    def apply_noise_electrons(
        self,
        image_e: u.Quantity,
        exposure: u.Quantity,
        rng: np.random.Generator,
    ) -> u.Quantity:
        image_e = u.Quantity(image_e).to(u.electron)
        exposure = u.Quantity(exposure).to(u.s)

        expected_e = image_e + self.dark_current * exposure
        expected_values = np.clip(expected_e.to_value(u.electron), 0, None)
        noisy_e = rng.poisson(expected_values).astype(float) * u.electron

        # Saturation occurs in each physical pixel before the detector is read.
        if self.full_well is not None:
            noisy_e = (
                np.clip(
                    noisy_e.to_value(u.electron),
                    0,
                    self.full_well.to_value(u.electron),
                )
                * u.electron
            )

        if self.read_noise.value > 0:
            noisy_e += (
                rng.normal(
                    0.0,
                    self.read_noise.to_value(u.electron),
                    size=noisy_e.shape,
                )
                * u.electron
            )

        return noisy_e

    def to_adu(self, image_e: u.Quantity) -> u.Quantity:
        image_e = u.Quantity(image_e).to(u.electron)
        return (image_e / self.gain).to(u.adu) + self.bias

    def apply_noise(
        self,
        image_e: u.Quantity,
        exposure: u.Quantity,
        rng: np.random.Generator,
    ) -> u.Quantity:
        noisy_e = self.apply_noise_electrons(image_e, exposure, rng)
        return self.to_adu(noisy_e)


@dataclass(frozen=True)
class DetectorReadout:
    detector: DetectorModel
    binning: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.binning, bool)
            or not isinstance(self.binning, (int, np.integer))
            or self.binning < 1
        ):
            raise ValueError("binning must be a positive integer.")

        binning = int(self.binning)
        if self.detector.nx % binning != 0 or self.detector.ny % binning != 0:
            raise ValueError(
                "binning must evenly divide both detector dimensions."
            )
        object.__setattr__(self, "binning", binning)

    @property
    def nx(self) -> int:
        return self.detector.nx // self.binning

    @property
    def ny(self) -> int:
        return self.detector.ny // self.binning

    @property
    def pixel_size(self) -> u.Quantity:
        return self.detector.pixel_size * self.binning

    @property
    def read_noise(self) -> u.Quantity:
        return self.detector.read_noise * self.binning

    @property
    def dark_current(self) -> u.Quantity:
        return self.detector.dark_current * self.binning**2

    @property
    def gain(self) -> u.Quantity:
        return self.detector.gain

    @property
    def bias(self) -> u.Quantity:
        return self.detector.bias

    @property
    def qe_resource(self) -> str | None:
        return self.detector.qe_resource

    @property
    def window_resource(self) -> str | None:
        return self.detector.window_resource

    def qe_curve(self) -> ThroughputCurve:
        return self.detector.qe_curve()

    def window_curve(self) -> ThroughputCurve:
        return self.detector.window_curve()

    def bin_electrons(self, image_e: u.Quantity) -> u.Quantity:
        image_e = u.Quantity(image_e).to(u.electron)
        expected_shape = (self.detector.ny, self.detector.nx)
        if image_e.shape != expected_shape:
            raise ValueError(
                "image_e must have the native detector shape "
                f"{expected_shape}; got {image_e.shape}."
            )

        if self.binning == 1:
            return image_e

        values = image_e.to_value(u.electron).reshape(
            self.ny,
            self.binning,
            self.nx,
            self.binning,
        )
        return values.sum(axis=(1, 3)) * u.electron

    def apply_noise(
        self,
        image_e: u.Quantity,
        exposure: u.Quantity,
        rng: np.random.Generator,
    ) -> u.Quantity:
        noisy_native_e = self.detector.apply_noise_electrons(
            image_e,
            exposure,
            rng,
        )
        binned_e = self.bin_electrons(noisy_native_e)
        return self.detector.to_adu(binned_e)


FLI_KL400 = DetectorModel(
    nx=2048,
    ny=2048,
    pixel_size=11 * u.micron,
    gain=0.478 * u.electron / u.adu,
    read_noise=1.6 * u.electron,
    dark_current=0.4 * u.electron / u.s,
    bias=200.0 * u.adu,
    full_well=90000.0 * u.electron,
    name="FLI Kepler KL400",
    qe_resource="gsense400bsi_qe",
)

FLI_AR571 = DetectorModel(
    nx=6244,
    ny=4168,
    pixel_size=3.76 * u.um,
    # gain=?,
    read_noise=1.0 * u.electron,
    dark_current=0.002 * u.electron / u.s,
    bias=200 * u.adu,
    full_well=50000.0 * u.electron,
    name="FLI Aurora AR571",
    qe_resource="AR571_qe",
    window_resource="UVFS_coating",
)

QHY_268M = DetectorModel(
    nx=6280,
    ny=4210,
    pixel_size=3.76 * u.micron,
    gain=0.52 * u.electron / u.adu,
    read_noise=2.18 * u.electron,
    dark_current=0.0005 * u.electron / u.s,
    # bias=?,
    full_well=43977 * u.electron,
    name="QHY268M",
    qe_resource="qhy268_qe",
)
