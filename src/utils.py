
from pathlib import Path
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table


class HeaderTable(Table):

    @property
    def header(self) -> dict:
        return self.meta


def read_reference_spectrum(filepath: Path):
    if "SNIFS" in filepath.name.upper():
        return read_snifs_spectrum(filepath)
    if filepath.suffix.lower() == ".ecsv":
        table = HeaderTable.read(filepath)
        table.meta["OBJECT"] = table.meta.get("OBJECT", table.meta.get("TARGETID"))
        if "coordinates" in table.meta:
            coord = table.meta["coordinates"]
            table.meta["RA"] = coord.ra.deg
            table.meta["DEC"] = coord.dec.deg
        return table


def read_snifs_spectrum(filepath: Path) -> fits.BinTableHDU:
    data = Table.read(filepath, format="ascii.commented_header", names=["wl", "fl", "err"])
    data["wl"].unit = "Angstrom"
    data["fl"].unit = data["err"].unit = "erg/s/cm2/Angstrom"
    hdu = fits.BinTableHDU(data=data, name="SPECTRUM")
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("#"):
                try:
                    key, val = line.lstrip("#").strip().split("=", 1)
                except:
                    continue
                hdu.header[key.replace("FLUXCAL_", "CAL")[:8]] = val.strip()
    for kw in ["EXPTIME", "AIRMASS", "MJD-OBS"]:
        if kw in hdu.header:
            hdu.header[kw] = float(hdu.header[kw].rstrip("s"))
    return hdu
