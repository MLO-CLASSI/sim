from dataclasses import dataclass

import numpy as np
from astropy import units as u
from shared_data import CSV_FILES

from .optics import ThroughputCurve


@dataclass(frozen=True)
class FiberModel:
    name: str
    attenuation_resource: str
    core_diameter: u.Quantity | None = None

    def __post_init__(self) -> None:
        if self.core_diameter is not None:
            object.__setattr__(
                self,
                "core_diameter",
                u.Quantity(self.core_diameter).to(u.um),
            )

    def throughput_curve(self, length: u.Quantity) -> ThroughputCurve:
        length = u.Quantity(length).to(u.m)
        if length < 0 * u.m:
            raise ValueError("Fiber length cannot be negative.")

        wavelength_nm, attenuation_db_per_km = np.loadtxt(
            CSV_FILES[self.attenuation_resource],
            delimiter=",",
        ).T
        transmission = 10 ** (
            -attenuation_db_per_km * length.to_value(u.km) / 10
        )
        return ThroughputCurve(
            wavelength_nm * u.nm,
            transmission,
            name=self.name,
        )


UVNS = FiberModel(
    name="UV(NS) fiber",
    attenuation_resource="uvns_attenuation",
)

UVWFS = FiberModel(
    name="UV(WF)S fiber",
    attenuation_resource="uvwfs_attenuation",
)

WF = FiberModel(
    name="WF fiber",
    attenuation_resource="wf_attenuation",
)

WFNS = FiberModel(
    name="WFNS fiber",
    attenuation_resource="wfns_attenuation",
)

HPSC25 = FiberModel(
    name="HPSC25 fiber",
    attenuation_resource="hpsc25_attenuation",
)

CLASSI_FIBER = FiberModel(
    name="CLASSI 105 um UV(NS) fiber",
    attenuation_resource="uvns_attenuation",
    core_diameter=105 * u.um,
)
