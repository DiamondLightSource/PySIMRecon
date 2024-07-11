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
    wavelength: int  # Emission wavelength
    otf: Path | None
    reconstruction_config: dict[str, Any] = field(default_factory=dict)
    otf_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SettingsManager:
    defaults_config_path: str | PathLike[str]
    default_reconstruction_config: dict[str, Any] = field(default_factory=dict)
    default_otf_config: dict[str, Any] = field(default_factory=dict)
    wavelength_settings: InitVar[Iterable[WavelengthSettings]] = field(
        default_factory=tuple
    )
    wavelengths: dict[int, WavelengthSettings] = field(default_factory=dict, init=False)

    def __post_init__(self, wavelength_settings) -> None:
        for settings in wavelength_settings:
            self.set_wavelength(settings)

    def set_wavelength(self, wavelength_settings: WavelengthSettings) -> None:
        self.wavelengths[wavelength_settings.wavelength] = wavelength_settings

    def get_wavelength(self, wavelength: int) -> WavelengthSettings | None:
        return self.wavelengths.get(wavelength, None)

    def get_reconstruction_config(
        self, wavelength: int, include_defaults: bool = True
    ) -> dict[str, Any]:
        if include_defaults:
            config = self.default_reconstruction_config.copy()
        else:
            config = {}
        config.update(self.get_wavelength(wavelength).reconstruction_config)
        return config

    def get_otf_config(
        self, wavelength: int, include_defaults: bool = True
    ) -> dict[str, Any]:
        if include_defaults:
            config = self.default_otf_config.copy()
        else:
            config = {}
        config.update(self.get_wavelength(wavelength).otf_config)
        return config

    def get_otf_path(self, wavelength: int) -> Path | None:
        return self.get_wavelength(wavelength).otf
