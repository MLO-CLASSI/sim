import numpy as np
from astropy import units as u
from shared_data import CSV_FILES

from .optics import ThroughputCurve


class AtmosphericExtinction(ThroughputCurve):
    def __init__(
        self,
        airmass: float = 1.0,
        name: str = "atmosphere",
        fill_value: float = 0.0,
    ):
        wavelength_nm, extinction_mag_per_airmass = np.loadtxt(
            CSV_FILES["palomar_atm_ext_per_airmass"],
            delimiter=",",
        ).T
        throughput = 10 ** (-0.4 * extinction_mag_per_airmass * airmass)
        super().__init__(
            wavelength_nm * u.nm,
            throughput,
            name=name,
            fill_value=fill_value,
        )
