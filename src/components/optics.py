
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units as u
from shared_data import CSV_FILES


@dataclass
class ThroughputCurve:
    wavelength: u.Quantity
    throughput: np.ndarray
    name: str = ""
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        self.wavelength = u.Quantity(self.wavelength).to(u.AA)
        self.throughput = u.Quantity(self.throughput).to_value(u.dimensionless_unscaled)
        if self.wavelength.shape != self.throughput.shape:
            raise ValueError("wavelength and throughput must have the same shape.")

    def __call__(self, wavelength: u.Quantity) -> np.ndarray:
        return np.interp(
            wavelength.to_value(u.AA),
            self.wavelength.to_value(u.AA),
            self.throughput,
            left=self.fill_value,
            right=self.fill_value,
        )

    @classmethod
    def from_csv(
        cls,
        fname,
        name: str = "",
        fill_value: float = 0.0,
        wavelength_unit: u.UnitBase = u.nm,
    ):
        df = pd.read_csv(fname, header=None, names=["wav", "tx"])
        return cls(
            df["wav"].values * wavelength_unit,
            df["tx"].values,
            name,
            fill_value,
        )


@dataclass
class OpticalElement:
    name: str
    throughput_resource: str

    def throughput_curve(self) -> ThroughputCurve:
        return ThroughputCurve.from_csv(
            CSV_FILES[self.throughput_resource],
            name=self.name,
        )


@dataclass
class FocalOptic(OpticalElement):
    focal_length: u.Quantity
    throughput_source_note: str = ""

    def __post_init__(self) -> None:
        self.focal_length = u.Quantity(self.focal_length).to(u.mm)


E02_PICKOFF = OpticalElement(
    name="E02 pickoff mirror",
    throughput_resource="e02_mirror_coating",
)

FGL400S = OpticalElement(
    name="Thorlabs FGL400S long-pass filter",
    throughput_resource="FGL400S_transmission",
)

UVFS_WINDOW = OpticalElement(
    name="UV fused-silica detector window",
    throughput_resource="UVFS_coating",
)

THORLABS_AC508_180_AB = FocalOptic(
    name="Thorlabs AC508-180-AB",
    focal_length=180 * u.mm,
    throughput_resource="ac508-180-ab",
)

CANON_EF100_F2 = FocalOptic(
    name="Canon EF 100 mm f/2 USM",
    focal_length=100 * u.mm,
    throughput_resource="LensTip_CanonEF85mm",
    throughput_source_note=(
        "Transmission curve is currently represented by the available "
        "LensTip Canon EF 85 mm data as a proxy."
    ),
)