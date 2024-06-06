from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from pathlib import Path
    from collections.abc import Iterable


@dataclass(slots=True)
class WavelengthSettings:
    wavelength: int
    otf: Path | None
    psf: Path | None
    config: dict[str, Any] = field(default_factory=dict)
    otf_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SettingsManager:
    defaults_config: str | PathLike[str]
    wavelengths_config: str | PathLike[str]
    wavelength_settings: InitVar[Iterable[WavelengthSettings]] = field(
        default_factory=tuple
    )
    wavelengths: dict[int, WavelengthSettings] = field(default_factory=dict, init=False)

    def __postinit__(self, wavelength_settings) -> None:
        for settings in wavelength_settings:
            self.set_wavelength(settings)

    def set_wavelength(self, wavelength_settings: WavelengthSettings) -> None:
        self.wavelengths[wavelength_settings.wavelength] = wavelength_settings

    def get_wavelength(self, wavelength: int) -> WavelengthSettings | None:
        return self.wavelengths.get(wavelength, None)
