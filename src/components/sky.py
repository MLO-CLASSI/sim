from dataclasses import dataclass

from astropy import units as u
from astropy.table import Table
from shared_data import REFERENCE_SPECTRA


SKY_SURFACE_BRIGHTNESS_UNIT = (
    u.erg / u.s / u.cm**2 / u.AA / u.arcsec**2
)


@dataclass(frozen=True)
class SkySpectrum:
    name: str
    spectrum_resource: str

    def spectrum(self) -> tuple[u.Quantity, u.Quantity]:
        table = Table.read(REFERENCE_SPECTRA[self.spectrum_resource])
        wavelength = table["wavelength"].quantity.to(u.AA)
        surface_brightness = table["flux"].quantity.to(
            SKY_SURFACE_BRIGHTNESS_UNIT
        )
        return wavelength, surface_brightness


DESI_SKY_DARK = SkySpectrum(
    name="DESI dark sky",
    spectrum_resource="desi_sky_dark",
)

DESI_SKY_GREY = SkySpectrum(
    name="DESI grey sky",
    spectrum_resource="desi_sky_grey",
)

DESI_SKY_BRIGHT = SkySpectrum(
    name="DESI bright sky",
    spectrum_resource="desi_sky_bright",
)


__all__ = [
    "DESI_SKY_BRIGHT",
    "DESI_SKY_DARK",
    "DESI_SKY_GREY",
    "SKY_SURFACE_BRIGHTNESS_UNIT",
    "SkySpectrum",
]
