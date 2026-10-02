import numpy as np
from astropy import units as u
from astropy.table import Table
from shared_data import CSV_FILES

from .optics import ThroughputCurve


class AtmosphericExtinction(ThroughputCurve):
    def __init__(
        self,
        airmass: float = 1.0,
        name: str = "atmosphere",
        fill_value: float = 0.0,
    ):
        lsst_atm = Table.read(CSV_FILES["atm_lsst"]) # airmass 1.0
        super().__init__(
            lsst_atm["Wavelength"].quantity,
            lsst_atm["Throughput"]**airmass,
            name=name,
            fill_value=fill_value,
        )
        self.airmass = airmass
