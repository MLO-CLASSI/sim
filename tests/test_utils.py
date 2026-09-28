from astropy import units as u
from numpy.testing import assert_allclose

from simulator import read_snifs_spectrum


def test_read_snifs_spectrum_preserves_data_units_and_metadata(tmp_path):
    path = tmp_path / "snifs.dat"
    path.write_text(
        "# EXPTIME=1200s\n"
        "# AIRMASS=1.23\n"
        "# MJD-OBS=61234.5\n"
        "# FLUXCAL_VERSION=test-cal\n"
        "# wl fl err\n"
        "4000 1.0e-15 1.0e-17\n"
        "5000 2.0e-15 2.0e-17\n"
    )

    hdu = read_snifs_spectrum(path)

    assert hdu.name == "SPECTRUM"
    assert u.Unit(hdu.columns["wl"].unit) == u.AA
    expected_flux_unit = u.erg / (u.s * u.cm**2 * u.AA)
    assert u.Unit(hdu.columns["fl"].unit).is_equivalent(expected_flux_unit)
    assert u.Unit(hdu.columns["err"].unit).is_equivalent(expected_flux_unit)
    assert_allclose(hdu.data["wl"], [4000.0, 5000.0])
    assert_allclose(hdu.data["fl"], [1.0e-15, 2.0e-15])
    assert hdu.header["EXPTIME"] == 1200.0
    assert hdu.header["AIRMASS"] == 1.23
    assert hdu.header["MJD-OBS"] == 61234.5
    assert hdu.header["CALVERSI"] == "test-cal"
