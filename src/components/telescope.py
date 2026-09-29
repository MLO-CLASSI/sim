from dataclasses import dataclass

import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class TelescopeModel:
    name: str
    diameter: u.Quantity
    focal_length: u.Quantity
    obstruction_diameter: u.Quantity = 0.0 * u.m

    def __post_init__(self) -> None:
        diameter = u.Quantity(self.diameter).to(u.m)
        focal_length = u.Quantity(self.focal_length).to(u.mm)
        obstruction_diameter = u.Quantity(self.obstruction_diameter).to(u.m)

        if diameter <= 0 * u.m:
            raise ValueError("diameter must be positive.")
        if focal_length <= 0 * u.mm:
            raise ValueError("focal_length must be positive.")
        if obstruction_diameter < 0 * u.m:
            raise ValueError("obstruction_diameter cannot be negative.")
        if obstruction_diameter >= diameter:
            raise ValueError("obstruction_diameter must be smaller than diameter.")

        object.__setattr__(self, "diameter", diameter)
        object.__setattr__(self, "focal_length", focal_length)
        object.__setattr__(self, "obstruction_diameter", obstruction_diameter)

    @property
    def collecting_area(self) -> u.Quantity:
        return (
            np.pi
            * (self.diameter**2 - self.obstruction_diameter**2)
            / 4
        ).to(u.m**2)

    @property
    def plate_scale(self) -> u.Quantity:
        return (
            (1 * u.rad).to(u.arcsec)
            / self.focal_length.to(u.mm)
        )


CLAUD_50INCH = TelescopeModel(
    name="Claud 1.25-m Telescope",
    diameter=1250*u.mm,
    focal_length=8125*u.mm,
    obstruction_diameter=375*u.mm,
)
