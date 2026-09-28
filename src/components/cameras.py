
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units as u


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

    def __post_init__(self) -> None:
        self.pixel_size = u.Quantity(self.pixel_size).to(u.um)
        self.gain = u.Quantity(self.gain).to(u.electron / u.adu)
        self.read_noise = u.Quantity(self.read_noise).to(u.electron)
        self.dark_current = u.Quantity(self.dark_current).to(u.electron / u.s)
        self.bias = u.Quantity(self.bias).to(u.adu)
        if self.full_well is not None:
            self.full_well = u.Quantity(self.full_well).to(u.electron)

    def apply_noise(
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

        if self.read_noise.value > 0:
            noisy_e += (
                rng.normal(
                    0.0,
                    self.read_noise.to_value(u.electron),
                    size=noisy_e.shape,
                )
                * u.electron
            )

        if self.full_well is not None:
            noisy_e = (
                np.clip(
                    noisy_e.to_value(u.electron),
                    0,
                    self.full_well.to_value(u.electron),
                )
                * u.electron
            )

        return (noisy_e / self.gain).to(u.adu) + self.bias

FLI_KL400 = DetectorModel(
    nx=2048,
    ny=2048,
    pixel_size=11*u.micron,
    gain=0.478*u.electron/u.adu,
    read_noise=1.6*u.electron,
    dark_current=0.4*u.electron/u.s,
    bias=200.0*u.adu,
    full_well=90000.0*u.electron,
)

FLI_AR571 = DetectorModel(
    nx=6244,
    ny=4168,
    pixel_size=3.76*u.um,
    # gain=?,
    read_noise=1.0*u.electron,
    dark_current=0.002*u.electron/u.s,
    bias=200*u.adu,
    full_well=50000.0*u.electron,
)

QHY_268M = DetectorModel(
    nx=6280,
    ny=4210,
    pixel_size=3.76*u.micron,
    gain=0.52*u.electron/u.adu,
    read_noise=2.18*u.electron,
    dark_current=0.0005*u.electron/u.s,
    # bias=?,
    full_well=43977*u.electron,
)
