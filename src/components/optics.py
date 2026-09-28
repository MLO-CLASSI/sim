
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy import units as u

@dataclass
class ThroughputCurve:
    wavelength: u.Quantity
    throughput: np.ndarray
    name: str = ""
    fill_value: float = 0.0

    def __post_init__(self) -> None:
        self.wavelength = u.Quantity(self.wavelength)
        if self.wavelength.unit == u.dimensionless_unscaled:
            raise u.UnitConversionError("Throughput wavelength values must have units.")

        throughput = u.Quantity(self.throughput)
        self.throughput = throughput.to_value(u.dimensionless_unscaled)
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

