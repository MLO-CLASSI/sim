
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from astropy import constants as const, units as u
from astropy.modeling import models
from specreduce.wavesol1d import WavelengthSolution1D

from .components import (
    AtmosphericExtinction,
    CLAUD_50INCH,
    DetectorModel,
    DetectorReadout,
    FiberModel,
    FocalOptic,
    OpticalElement,
    GratingModel,
    SkySpectrum,
    TelescopeModel,
    ThroughputCurve,
)

FLUX_DENSITY_UNIT = u.erg / u.s / u.cm**2 / u.AA
WAVELENGTH_SOLUTION_DEGREE = 4


@dataclass
class SpectrographModel:
    detector: DetectorModel | DetectorReadout
    groove_density: u.Quantity
    incidence_angle: u.Quantity
    diffraction_angle: u.Quantity
    collimator_focal_length: u.Quantity
    camera_focal_length: u.Quantity
    fiber_core_diameter: u.Quantity
    diffraction_order: int = 1
    fiber_count: int = 1
    fiber_pitch: u.Quantity = 250.0 * u.um
    wavelength_increases_with_x: bool = False
    kernel_radius_sigma: float = 4.0
    render_sampling_px: float = 0.5

    trace_func: Callable[[u.Quantity, u.Quantity], u.Quantity] | None = None

    grating: GratingModel | None = None
    collimator: FocalOptic | None = None
    camera_lens: FocalOptic | None = None
    fiber: FiberModel | None = None
    optical_elements: tuple[OpticalElement, ...] = ()
    extra_throughputs: tuple[ThroughputCurve, ...] = ()

    @classmethod
    def from_components(
        cls,
        *,
        detector: DetectorModel,
        grating: GratingModel,
        collimator: FocalOptic,
        camera_lens: FocalOptic,
        fiber: FiberModel,
        incidence_angle: u.Quantity,
        diffraction_angle: u.Quantity,
        diffraction_order: int = 1,
        fiber_count: int = 1,
        fiber_pitch: u.Quantity = 250.0 * u.um,
        wavelength_increases_with_x: bool = False,
        kernel_radius_sigma: float = 4.0,
        render_sampling_px: float = 0.5,
        trace_func: Callable[[u.Quantity, u.Quantity], u.Quantity] | None = None,
        optical_elements: tuple[OpticalElement, ...] = (),
        extra_throughputs: tuple[ThroughputCurve, ...] = (),
    ) -> "SpectrographModel":
        if fiber.core_diameter is None:
            raise ValueError(
                "fiber.core_diameter must be defined to construct a spectrograph."
            )

        return cls(
            detector=detector,
            groove_density=grating.groove_density,
            incidence_angle=incidence_angle,
            diffraction_angle=diffraction_angle,
            collimator_focal_length=collimator.focal_length,
            camera_focal_length=camera_lens.focal_length,
            fiber_core_diameter=fiber.core_diameter,
            diffraction_order=diffraction_order,
            fiber_count=fiber_count,
            fiber_pitch=fiber_pitch,
            wavelength_increases_with_x=wavelength_increases_with_x,
            kernel_radius_sigma=kernel_radius_sigma,
            render_sampling_px=render_sampling_px,
            trace_func=trace_func,
            grating=grating,
            collimator=collimator,
            camera_lens=camera_lens,
            fiber=fiber,
            optical_elements=optical_elements,
            extra_throughputs=extra_throughputs,
        )

    def __post_init__(self) -> None:
        self.groove_density = u.Quantity(self.groove_density).to(1 / u.mm)
        self.incidence_angle = u.Quantity(self.incidence_angle).to(u.deg)
        self.diffraction_angle = u.Quantity(self.diffraction_angle).to(u.deg)
        self.collimator_focal_length = u.Quantity(
            self.collimator_focal_length
        ).to(u.mm)
        self.camera_focal_length = u.Quantity(self.camera_focal_length).to(u.mm)
        self.fiber_core_diameter = u.Quantity(self.fiber_core_diameter).to(u.um)
        self.fiber_pitch = u.Quantity(self.fiber_pitch).to(u.um)
        self.optical_elements = tuple(self.optical_elements)
        self.extra_throughputs = tuple(self.extra_throughputs)

        if self.diffraction_order == 0:
            raise ValueError("diffraction_order must be non-zero.")
        if self.fiber_count < 1:
            raise ValueError("fiber_count must be at least 1.")
        if self.render_sampling_px <= 0:
            raise ValueError("render_sampling_px must be positive.")

    def throughput_curves(self) -> list[ThroughputCurve]:
        curves = [
            element.throughput_curve()
            for element in self.optical_elements
        ]

        if self.fiber is not None:
            curves.append(self.fiber.throughput_curve())
        if self.collimator is not None:
            curves.append(self.collimator.throughput_curve())
        if self.grating is not None:
            curves.append(self.grating.throughput_curve())
        if self.camera_lens is not None:
            curves.append(self.camera_lens.throughput_curve())
        if self.detector.window_resource is not None:
            curves.append(self.detector.window_curve())
        if self.detector.qe_resource is not None:
            curves.append(self.detector.qe_curve())

        curves.extend(self.extra_throughputs)
        return curves

    @property
    def groove_spacing(self) -> u.Quantity:
        return (1 / self.groove_density).to(u.mm)

    @property
    def central_wavelength(self) -> u.Quantity:
        wavelength = (
            self.groove_spacing
            * (
                np.sin(self.incidence_angle)
                + np.sin(self.diffraction_angle)
            )
            / self.diffraction_order
        )
        return wavelength.to(u.AA)

    @property
    def dispersion(self) -> u.Quantity:
        dispersion = (
            self.groove_spacing
            * np.cos(self.diffraction_angle)
            / (self.diffraction_order * self.camera_focal_length)
            * self.detector.pixel_size
            / u.pixel
        )
        return dispersion.to(u.AA / u.pixel)

    @property
    def x_center(self) -> u.Quantity:
        return (self.detector.nx - 1) / 2 * u.pixel

    @property
    def trace_y(self) -> u.Quantity:
        return (self.detector.ny - 1) / 2 * u.pixel

    @property
    def magnification(self) -> u.Quantity:
        return (
            self.camera_focal_length / self.collimator_focal_length
        ).to(u.dimensionless_unscaled)

    @property
    def anamorphic_factor(self) -> u.Quantity:
        return (
            np.cos(self.incidence_angle) / np.cos(self.diffraction_angle)
        ).to(u.dimensionless_unscaled)

    @property
    def fiber_pitch_px(self) -> u.Quantity:
        pitch_pixels = (
            self.fiber_pitch
            * self.magnification
            / self.detector.pixel_size
        ).to_value(u.dimensionless_unscaled)
        return pitch_pixels * u.pixel

    @property
    def spatial_fwhm_px(self) -> u.Quantity:
        width_pixels = (
            self.fiber_core_diameter
            * self.magnification
            / self.detector.pixel_size
        ).to_value(u.dimensionless_unscaled)
        return width_pixels * u.pixel

    @property
    def spectral_fwhm_px(self) -> u.Quantity:
        return self.spatial_fwhm_px * self.anamorphic_factor

    @property
    def spatial_sigma_px(self) -> u.Quantity:
        return self.spatial_fwhm_px / (2 * np.sqrt(2 * np.log(2)))

    @property
    def spectral_sigma_px(self) -> u.Quantity:
        return self.spectral_fwhm_px / (2 * np.sqrt(2 * np.log(2)))

    def _exact_wavelength_for_pixel(self, pixel) -> np.ndarray:
        pixel = np.asarray(pixel, dtype=float)
        center = self.x_center.to_value(u.pixel)
        if self.wavelength_increases_with_x:
            detector_offset = pixel - center
        else:
            detector_offset = center - pixel

        field_angle = np.arctan(
            detector_offset
            * (self.detector.pixel_size / self.camera_focal_length).to_value(
                u.dimensionless_unscaled
            )
        ) * u.rad
        diffraction_angle = self.diffraction_angle + field_angle
        wavelength = (
            self.groove_spacing
            * (
                np.sin(self.incidence_angle)
                + np.sin(diffraction_angle)
            )
            / self.diffraction_order
        )

        return wavelength.to_value(u.AA)

    @cached_property
    def wavelength_solution(self) -> WavelengthSolution1D:
        """Pixel-to-wavelength solution for this detector sampling."""
        pixels = np.arange(self.detector.nx, dtype=float)
        center = self.x_center.to_value(u.pixel)
        wavelength = self._exact_wavelength_for_pixel(pixels)
        coefficients = np.polynomial.polynomial.polyfit(
            pixels - center,
            wavelength,
            WAVELENGTH_SOLUTION_DEGREE,
        )
        polynomial = models.Polynomial1D(
            WAVELENGTH_SOLUTION_DEGREE,
            **{
                f"c{order}": coefficient
                for order, coefficient in enumerate(coefficients)
            },
        )
        pixel_to_wavelength = models.Shift(-center) | polynomial
        return WavelengthSolution1D(
            p2w=pixel_to_wavelength,
            bounds_pix=(0, self.detector.nx),
            unit=u.AA,
        )

    def wavelength_to_x(self, wavelength: u.Quantity) -> u.Quantity:
        """Map wavelength to detector pixel using ``wavelength_solution``."""
        wavelength = u.Quantity(wavelength).to(self.groove_spacing.unit)
        sin_diffraction_angle = (
            self.diffraction_order * wavelength / self.groove_spacing
            - np.sin(self.incidence_angle)
        ).to_value(u.dimensionless_unscaled)
        if np.any(np.abs(sin_diffraction_angle) > 1):
            raise ValueError(
                "At least one wavelength is not physically reachable for the "
                "configured grating geometry."
            )

        pixel = self.wavelength_solution.wav_to_pix(
            wavelength.to_value(self.wavelength_solution.unit)
        )
        return np.asarray(pixel) * u.pixel

    def x_to_wavelength(self, x: u.Quantity) -> u.Quantity:
        """Map detector pixel to wavelength using ``wavelength_solution``."""
        x = u.Quantity(x, u.pixel)
        wavelength = self.wavelength_solution.pix_to_wav(x.to_value(u.pixel))
        return np.asarray(wavelength) * self.wavelength_solution.unit

    def fiber_trace_centers(self) -> u.Quantity:
        offsets = (
            np.arange(self.fiber_count)
            - (self.fiber_count - 1) / 2
        )
        return self.trace_y + offsets * self.fiber_pitch_px

    def wavelength_to_y(
        self,
        wavelength: u.Quantity,
        x: u.Quantity,
        fiber_trace_y: u.Quantity | None = None,
    ) -> u.Quantity:
        if fiber_trace_y is None:
            fiber_trace_y = self.trace_y
        if self.trace_func is None:
            return np.full(x.shape, fiber_trace_y.to_value(u.pixel)) * u.pixel

        return fiber_trace_y + u.Quantity(
            self.trace_func(wavelength, x)
        ).to(u.pixel)


class InstrumentSimulator:
    def __init__(
        self,
        spectrograph: SpectrographModel,
        telescope: TelescopeModel = CLAUD_50INCH,
        atmosphere: AtmosphericExtinction | None = None,
        sky: SkySpectrum | None = None,
        *,
        binning: int = 1,
        throughputs: list[ThroughputCurve] | None = None,
    ) -> None:
        # Accept the previous InstrumentSimulator(spectrograph, throughputs)
        # form so downstream callers can migrate independently.
        if isinstance(telescope, list):
            if atmosphere is not None or throughputs is not None:
                raise TypeError(
                    "Legacy positional throughputs cannot be combined with "
                    "atmosphere or throughputs=."
                )
            throughputs = telescope
            telescope = CLAUD_50INCH

        if not isinstance(telescope, TelescopeModel):
            raise TypeError("telescope must be a TelescopeModel.")
        if isinstance(spectrograph.detector, DetectorReadout):
            raise TypeError(
                "spectrograph.detector must describe the native detector; "
                "set binning on InstrumentSimulator instead."
            )

        self.spectrograph = spectrograph
        self.detector = spectrograph.detector
        self.readout = DetectorReadout(self.detector, binning=binning)
        self.binning = self.readout.binning
        self.readout_spectrograph = replace(
            spectrograph,
            detector=self.readout,
        )
        self.telescope = telescope
        self.atmosphere = atmosphere
        self.sky = sky

        if throughputs is not None:
            if atmosphere is not None:
                raise ValueError(
                    "atmosphere cannot be combined with an explicit "
                    "throughputs override."
                )
            self.throughputs = list(throughputs)
            return

        self.throughputs = []
        if atmosphere is not None:
            self.throughputs.append(atmosphere)
        self.throughputs.extend(spectrograph.throughput_curves())

    def combined_throughput(
        self,
        wavelength: u.Quantity,
        *,
        include_atmosphere: bool = True,
    ) -> np.ndarray:
        wavelength = u.Quantity(wavelength)
        throughput = np.ones(wavelength.shape, dtype=float)

        for curve in self.throughputs:
            if (
                not include_atmosphere
                and isinstance(curve, AtmosphericExtinction)
            ):
                continue
            throughput *= curve(wavelength)

        return throughput

    @property
    def wavelength_solution(self) -> WavelengthSolution1D:
        """Wavelength solution for the configured detector readout."""
        return self.readout_spectrograph.wavelength_solution

    @property
    def fiber_sky_area(self) -> u.Quantity:
        angular_radius = (
            0.5
            * self.spectrograph.fiber_core_diameter
            / self.telescope.focal_length
        ).decompose().value * u.rad
        angular_radius = angular_radius.to(u.arcsec)
        return np.pi * angular_radius**2

    def render_electrons(
        self,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        exposure: u.Quantity,
        vignetting=None,
        fiber_coupling_efficiency=1.0,
    ) -> u.Quantity:
        native_image = self._render_native_electrons(
            wavelength=wavelength,
            flux_density=flux_density,
            exposure=exposure,
            vignetting=vignetting,
            fiber_coupling_efficiency=fiber_coupling_efficiency,
        )
        return self.readout.bin_electrons(native_image)

    def _render_native_electrons(
        self,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        exposure: u.Quantity,
        vignetting=None,
        fiber_coupling_efficiency=1.0,
    ) -> u.Quantity:
        wavelength = u.Quantity(wavelength).to(u.AA)
        flux_density = u.Quantity(flux_density).to(FLUX_DENSITY_UNIT)
        exposure = u.Quantity(exposure).to(u.s)

        if wavelength.ndim != 1:
            raise ValueError("wavelength must be a 1D array.")

        if flux_density.ndim == 1:
            if self.spectrograph.fiber_count != 1:
                raise ValueError(
                    "flux_density must have shape (fiber_count, n_wavelength) "
                    "when simulating multiple fibers."
                )
            flux_density = flux_density[np.newaxis, :]
        elif flux_density.ndim != 2:
            raise ValueError(
                "flux_density must be a 1D array for one fiber or a 2D array "
                "with shape (fiber_count, n_wavelength)."
            )

        if flux_density.shape[0] != self.spectrograph.fiber_count:
            raise ValueError(
                "flux_density must contain one spectrum per fiber: "
                f"expected {self.spectrograph.fiber_count}, "
                f"got {flux_density.shape[0]}."
            )

        if wavelength.size != flux_density.shape[1]:
            raise ValueError(
                "wavelength length must match the number of wavelength samples "
                "in each input spectrum."
            )

        order = np.argsort(wavelength.value)
        wavelength = wavelength[order]
        flux_density = flux_density[:, order]

        wavelength, flux_density = self._resample_for_detector(
            wavelength,
            flux_density,
        )

        throughput = self.combined_throughput(wavelength)
        d_wavelength = self._trapezoid_bin_widths(wavelength)
        photon_energy = (const.h * const.c / wavelength).to(u.erg)

        expected_electrons = (
            flux_density
            * self.telescope.collecting_area
            * d_wavelength
            * exposure
            * throughput
            / photon_energy
        ).to_value(u.dimensionless_unscaled) * u.electron

        expected_electrons = (
            np.clip(expected_electrons.to_value(u.electron), 0, None)
            * u.electron
        )

        coupling = np.asarray(
            u.Quantity(fiber_coupling_efficiency).to_value(
                u.dimensionless_unscaled
            ),
            dtype=float,
        )
        if coupling.ndim == 0:
            if not 0 <= coupling.item() <= 1:
                raise ValueError(
                    "fiber_coupling_efficiency must be between 0 and 1."
                )
            expected_electrons *= coupling.item()
        elif coupling.shape == (self.spectrograph.fiber_count,):
            if np.any((coupling < 0) | (coupling > 1)):
                raise ValueError(
                    "fiber_coupling_efficiency values must be between 0 and 1."
                )
            expected_electrons *= coupling[:, np.newaxis]
        else:
            raise ValueError(
                "fiber_coupling_efficiency must be a scalar or contain one "
                "value per fiber."
            )

        x_centers = self.spectrograph.wavelength_to_x(wavelength)
        image = (
            np.zeros((self.detector.ny, self.detector.nx), dtype=float)
            * u.electron
        )

        for fiber_trace_y, fiber_bin_electrons in zip(
            self.spectrograph.fiber_trace_centers(),
            expected_electrons,
        ):
            y_centers = self.spectrograph.wavelength_to_y(
                wavelength,
                x_centers,
                fiber_trace_y,
            )

            self._deposit_gaussian_packets(
                image=image,
                x_centers=x_centers,
                y_centers=y_centers,
                counts=fiber_bin_electrons,
                sigma_x=self.spectrograph.spectral_sigma_px,
                sigma_y=self.spectrograph.spatial_sigma_px,
                radius_sigma=self.spectrograph.kernel_radius_sigma,
            )

        if self.sky is not None:
            image += self._render_native_sky_electrons(exposure)

        return self._apply_vignetting(image, vignetting)

    def render_sky_electrons(self, exposure: u.Quantity) -> u.Quantity:
        return self.readout.bin_electrons(
            self._render_native_sky_electrons(exposure)
        )

    def _render_native_sky_electrons(
        self,
        exposure: u.Quantity,
    ) -> u.Quantity:
        image = (
            np.zeros((self.detector.ny, self.detector.nx), dtype=float)
            * u.electron
        )
        if self.sky is None:
            return image

        exposure = u.Quantity(exposure).to(u.s)
        wavelength, surface_brightness = self.sky.spectrum()
        flux_density = (
            surface_brightness * self.fiber_sky_area
        ).to(FLUX_DENSITY_UNIT)

        x_centers = self.spectrograph.wavelength_to_x(wavelength)
        x_values = x_centers.to_value(u.pixel)
        radius_x = int(
            np.ceil(
                self.spectrograph.kernel_radius_sigma
                * self.spectrograph.spectral_sigma_px.to_value(u.pixel)
            )
        )
        on_detector = (
            (x_values >= -radius_x)
            & (x_values <= self.detector.nx - 1 + radius_x)
        )
        wavelength = wavelength[on_detector]
        flux_density = flux_density[on_detector]
        x_values = x_values[on_detector]

        if wavelength.size < 2:
            return image

        d_wavelength = self._trapezoid_bin_widths(wavelength)
        photon_energy = (const.h * const.c / wavelength).to(u.erg)
        throughput = self.combined_throughput(
            wavelength,
            include_atmosphere=False,
        )
        counts = (
            flux_density
            * self.telescope.collecting_area
            * d_wavelength
            * exposure
            * throughput
            / photon_energy
        ).to_value(u.dimensionless_unscaled)
        counts = np.clip(counts, 0, None)

        # DESI sky spectra are sampled much more finely than the detector needs.
        # Sum photon packets in detector-space bins so narrow lines retain their
        # integrated flux while avoiding one Gaussian deposition per 0.1-A sample.
        step = self.spectrograph.render_sampling_px
        bin_index = np.floor(x_values / step).astype(np.int64)
        _, inverse = np.unique(bin_index, return_inverse=True)
        binned_counts = np.bincount(inverse, weights=counts)
        weighted_x = np.bincount(inverse, weights=x_values * counts)
        weighted_wavelength = np.bincount(
            inverse,
            weights=wavelength.to_value(u.AA) * counts,
        )
        positive = binned_counts > 0
        binned_counts = binned_counts[positive]
        binned_x = weighted_x[positive] / binned_counts
        binned_wavelength = (
            weighted_wavelength[positive] / binned_counts
        ) * u.AA

        for fiber_trace_y in self.spectrograph.fiber_trace_centers():
            x = binned_x * u.pixel
            y = self.spectrograph.wavelength_to_y(
                binned_wavelength,
                x,
                fiber_trace_y,
            )
            self._deposit_gaussian_packets(
                image=image,
                x_centers=x,
                y_centers=y,
                counts=binned_counts * u.electron,
                sigma_x=self.spectrograph.spectral_sigma_px,
                sigma_y=self.spectrograph.spatial_sigma_px,
                radius_sigma=self.spectrograph.kernel_radius_sigma,
            )

        return image

    def _resample_for_detector(
        self,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
    ) -> tuple[u.Quantity, u.Quantity]:
        if wavelength.size < 2:
            raise ValueError("At least two wavelength samples are required.")

        wavelength_steps = np.diff(wavelength)
        if np.any(wavelength_steps <= 0 * wavelength.unit):
            raise ValueError("wavelength samples must be unique.")

        x = self.spectrograph.wavelength_to_x(wavelength).to_value(u.pixel)
        x_steps = np.abs(np.diff(x))
        if np.all(x_steps <= self.spectrograph.render_sampling_px):
            return wavelength, flux_density

        span_px = abs(x[-1] - x[0])
        uniform_count = (
            int(
                np.ceil(
                    span_px / self.spectrograph.render_sampling_px
                )
            )
            + 1
        )
        uniform_x = np.linspace(x[0], x[-1], uniform_count) * u.pixel
        uniform_wavelength = self.spectrograph.x_to_wavelength(
            uniform_x
        ).to(wavelength.unit)

        render_wavelength = (
            np.unique(
                np.concatenate(
                    (
                        wavelength.value,
                        uniform_wavelength.value,
                    )
                )
            )
            * wavelength.unit
        )
        render_flux_density = (
            np.vstack([
                np.interp(
                    render_wavelength.value,
                    wavelength.value,
                    spectrum.to_value(FLUX_DENSITY_UNIT),
                )
                for spectrum in flux_density
            ])
            * FLUX_DENSITY_UNIT
        )

        return render_wavelength, render_flux_density

    @staticmethod
    def _trapezoid_bin_widths(wavelength: u.Quantity) -> u.Quantity:
        values = wavelength.value
        widths = np.empty_like(values, dtype=float)
        widths[0] = 0.5 * (values[1] - values[0])
        widths[-1] = 0.5 * (values[-1] - values[-2])
        widths[1:-1] = 0.5 * (values[2:] - values[:-2])
        return widths * wavelength.unit

    def simulate(
        self,
        wavelength: u.Quantity,
        flux_density: u.Quantity,
        exposure: u.Quantity,
        vignetting=None,
        add_noise: bool = True,
        seed: int = None,
        fiber_coupling_efficiency=1.0,
    ) -> u.Quantity:
        native_image_e = self._render_native_electrons(
            wavelength=wavelength,
            flux_density=flux_density,
            exposure=exposure,
            vignetting=vignetting,
            fiber_coupling_efficiency=fiber_coupling_efficiency,
        )

        if not add_noise:
            return self.readout.bin_electrons(native_image_e)

        rng = np.random.default_rng(seed)
        return self.readout.apply_noise(
            image_e=native_image_e,
            exposure=exposure,
            rng=rng,
        )

    @staticmethod
    def _deposit_gaussian_packets(
        image: u.Quantity,
        x_centers: u.Quantity,
        y_centers: u.Quantity,
        counts: u.Quantity,
        sigma_x: u.Quantity,
        sigma_y: u.Quantity,
        radius_sigma: float,
    ) -> None:
        ny, nx = image.shape
        x_centers = u.Quantity(x_centers).to_value(u.pixel)
        y_centers = u.Quantity(y_centers).to_value(u.pixel)
        sigma_x = u.Quantity(sigma_x).to_value(u.pixel)
        sigma_y = u.Quantity(sigma_y).to_value(u.pixel)

        radius_x = int(np.ceil(radius_sigma * sigma_x))
        radius_y = int(np.ceil(radius_sigma * sigma_y))

        for x0, y0, count in zip(x_centers, y_centers, counts):
            if count.value <= 0:
                continue

            ix0 = int(np.floor(x0))
            iy0 = int(np.floor(y0))

            x_idx_full = np.arange(ix0 - radius_x, ix0 + radius_x + 1)
            y_idx_full = np.arange(iy0 - radius_y, iy0 + radius_y + 1)

            x_weight = np.exp(-0.5 * ((x_idx_full - x0) / sigma_x) ** 2)
            y_weight = np.exp(-0.5 * ((y_idx_full - y0) / sigma_y) ** 2)

            kernel_full = np.outer(y_weight, x_weight)
            kernel_sum = kernel_full.sum()

            if kernel_sum <= 0:
                continue

            x_ok = (x_idx_full >= 0) & (x_idx_full < nx)
            y_ok = (y_idx_full >= 0) & (y_idx_full < ny)

            if not np.any(x_ok) or not np.any(y_ok):
                continue

            x_idx = x_idx_full[x_ok]
            y_idx = y_idx_full[y_ok]
            kernel = kernel_full[np.ix_(y_ok, x_ok)]

            image[np.ix_(y_idx, x_idx)] += count * kernel / kernel_sum

    @staticmethod
    def _apply_vignetting(image: u.Quantity, vignetting) -> u.Quantity:
        if vignetting is None:
            return image

        if callable(vignetting):
            yy, xx = np.indices(image.shape)
            factor = vignetting(yy, xx)
        else:
            factor = vignetting

        factor = u.Quantity(factor).to_value(u.dimensionless_unscaled)
        if factor.shape != image.shape:
            raise ValueError("vignetting must have the same shape as the detector image.")

        return image * np.clip(factor, 0, None)

def f_lambda_to_photon_flux_density(
    wavelength: u.Quantity,
    flux_density: u.Quantity,
    collecting_area: u.Quantity,
) -> u.Quantity:
    wavelength = u.Quantity(wavelength).to(u.AA)
    flux_density = u.Quantity(flux_density).to(FLUX_DENSITY_UNIT)
    collecting_area = u.Quantity(collecting_area).to(u.cm**2)
    photon_energy = (const.h * const.c / wavelength).to(u.erg)

    return (flux_density * collecting_area / photon_energy).to(1 / u.s / u.AA)
