from __future__ import annotations
from dataclasses import dataclass, field, InitVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from pathlib import Path
    from collections.abc import Iterable

_REQUIRED_DEFAULTS = {
    "nphases": 5,
    "ndirs": 3,
    "zoomfact": 2,
    "zzoom": 1,
}


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
    wavelength_settings: InitVar[Iterable[WavelengthSettings]] = tuple()
    wavelengths: dict[int, WavelengthSettings] = field(init=False)

    def __post_init__(self, wavelength_settings: Iterable[WavelengthSettings]) -> None:
        self.wavelengths = {}
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
            config = _REQUIRED_DEFAULTS.copy()
            config.update(self.default_reconstruction_config)
        else:
            config = {}
        ws = self.get_wavelength(wavelength)
        if ws is not None:
            config.update(ws.reconstruction_config)
        return config

    def get_otf_config(
        self, wavelength: int, include_defaults: bool = True
    ) -> dict[str, Any]:
        if include_defaults:
            config = self.default_otf_config.copy()
        else:
            config = {}
        ws = self.get_wavelength(wavelength)
        if ws is not None:
            config.update(ws.otf_config)
        return config

    def get_otf_path(self, wavelength: int) -> Path | None:
        ws = self.get_wavelength(wavelength)
        if ws is None:
            raise ValueError(f"No settings for wavelength {wavelength}")
        return ws.otf
