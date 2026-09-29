from dataclasses import dataclass

import numpy as np
from astropy import units as u
from shared_data import CSV_FILES

from .optics import ThroughputCurve


@dataclass(frozen=True)
class GratingModel:
    name: str
    groove_density: u.Quantity
    efficiency_resources: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "groove_density",
            u.Quantity(self.groove_density).to(1 / u.mm),
        )
        if not self.efficiency_resources:
            raise ValueError("At least one efficiency resource is required.")

    def throughput_curve(self) -> ThroughputCurve:
        if len(self.efficiency_resources) == 1:
            return ThroughputCurve.from_csv(
                CSV_FILES[self.efficiency_resources[0]],
                name=self.name,
            )

        curves = []
        for resource in self.efficiency_resources:
            wavelength_nm, efficiency = np.loadtxt(
                CSV_FILES[resource],
                delimiter=",",
            ).T
            curves.append((wavelength_nm, efficiency))

        wave_min = min(wavelength.min() for wavelength, _ in curves)
        wave_max = max(wavelength.max() for wavelength, _ in curves)
        wavelength_nm = np.arange(np.floor(wave_min), np.ceil(wave_max) + 1)
        interpolated = [
            np.interp(
                wavelength_nm,
                wavelength,
                efficiency,
                left=np.nan,
                right=np.nan,
            )
            for wavelength, efficiency in curves
        ]
        efficiency = np.nanmean(np.vstack(interpolated), axis=0)
        efficiency = np.nan_to_num(efficiency, nan=0.0)

        return ThroughputCurve(
            wavelength_nm * u.nm,
            efficiency,
            name=self.name,
        )


NEWPORT_MASTER_1229 = GratingModel(
    name="Newport 270R master 1229",
    groove_density=300 / u.mm,
    efficiency_resources=(
        "master 1229 P plane",
        "master 1229 S plane",
    ),
)

NEWPORT_MASTER_1294 = GratingModel(
    name="Newport 270R master 1294",
    groove_density=300 / u.mm,
    efficiency_resources=("master 1294 unpolarized",),
)

THORLABS_GR50A_0305 = GratingModel(
    name="Thorlabs GR50-0305",
    groove_density=300 / u.mm,
    efficiency_resources=("gr50a-0305_efficiency-780",),
)
