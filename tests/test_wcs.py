import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from numpy.testing import assert_allclose

from simulator import LongSlitWCS


def make_imx571_wcs():
    return LongSlitWCS.from_instrument(
        ra_deg=120.0,
        dec_deg=22.0,
        position_angle_deg=35.0,
        nx=6244,
        ny=4168,
        pixel_size_mm=0.00376,
        groove_density_per_mm=300.0,
        collimator_focal_length_mm=180.0,
        camera_focal_length_mm=100.0,
        slit_width_mm=0.105,
        central_wavelength_nm=626.3304030251207,
        telescope_focal_length_mm=8125.0,
        incidence_angle_deg=32.0,
        diffraction_order=1,
        wavelength_increases_with_x=False,
    )


def test_wcs_reference_geometry():
    wcs = make_imx571_wcs()

    assert_allclose(wcs.reference_pixel, (3121.5, 2083.5), atol=1e-12)
    assert_allclose(wcs.diffraction_angle_deg, -20.0, atol=1e-10)
    assert_allclose(wcs.dispersion_nm_per_pixel, -0.11777480847183388, rtol=1e-12)
    assert_allclose(wcs.spatial_scale_arcsec_per_pixel, 0.17181541029911976, rtol=1e-12)
    assert_allclose(wcs.slit_width_arcsec, 2.6655759576547835, rtol=1e-12)
    assert_allclose(wcs.projected_slit_width_pixels, 14.001147024470933, rtol=1e-12)


def test_wcs_header_uses_cd_matrix_only():
    wcs = make_imx571_wcs()
    header = wcs.to_header()

    assert "CD1_1" in header
    assert "CD3_1" in header
    assert not any(key.startswith("PC") for key in header)
    assert not any(key.startswith("CDELT") for key in header)


def test_wcs_to_fits_serializes_a_readable_header():
    wcs = make_imx571_wcs()

    hdul = wcs.to_fits()
    header = hdul[0].header
    round_trip = WCS(header, preserve_units=True)

    assert "CD3_1" in header
    assert round_trip.pixel_n_dim == 3
    assert round_trip.world_n_dim == 3
    assert u.Unit(round_trip.wcs.cunit[2]).is_equivalent(u.m)


def test_wcs_reference_pixel_has_requested_world_coordinate():
    wcs = make_imx571_wcs()
    parent = wcs.parent_wcs
    x, y = wcs.reference_pixel

    ra, dec, wavelength = parent.pixel_to_world_values(x, y, 0.0)

    assert_allclose(ra, 120.0, atol=1e-10)
    assert_allclose(dec, 22.0, atol=1e-10)
    spectral_unit = u.Unit(parent.world_axis_units[2])
    assert_allclose((wavelength * spectral_unit).to_value(u.nm), 626.3304030251207, rtol=0, atol=1e-10)
