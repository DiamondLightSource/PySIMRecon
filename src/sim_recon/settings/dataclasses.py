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
class ChannelConfig:
    emission_wavelength_nm: int  # Emission wavelength
    # cudasirecon only accepts nanometre integer wavelengths, so channels
    # settings are limited in line with this.
    otf: Path | None
    reconstruction_config: dict[str, Any] = field(default_factory=dict)
    otf_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConfigManager:
    defaults_config_path: str | PathLike[str]
    default_reconstruction_config: dict[str, Any] = field(default_factory=dict)
    default_otf_config: dict[str, Any] = field(default_factory=dict)
    channel_configs: InitVar[Iterable[ChannelConfig]] = tuple()
    channels: dict[int, ChannelConfig] = field(init=False)

    def __post_init__(self, channel_configs: Iterable[ChannelConfig]) -> None:
        self.channels = {}
        for channel_config in channel_configs:
            self.set_channel_config(channel_config)

    def set_channel_config(self, wavelength_settings: ChannelConfig) -> None:
        self.channels[wavelength_settings.emission_wavelength_nm] = wavelength_settings

    def get_channel_config(self, emission_wavelength_nm: int) -> ChannelConfig | None:
        return self.channels.get(emission_wavelength_nm, None)

    def get_reconstruction_config(
        self, emission_wavelength_nm: int, include_defaults: bool = True
    ) -> dict[str, Any]:
        if include_defaults:
            config = _REQUIRED_DEFAULTS.copy()
            config.update(self.default_reconstruction_config)
        else:
            config = {}
        ws = self.get_channel_config(emission_wavelength_nm)
        if ws is not None:
            config.update(ws.reconstruction_config)
        return config

    def get_otf_config(
        self, emission_wavelength_nm: int, include_defaults: bool = True
    ) -> dict[str, Any]:
        if include_defaults:
            config = self.default_otf_config.copy()
        else:
            config = {}
        ws = self.get_channel_config(emission_wavelength_nm)
        if ws is not None:
            config.update(ws.otf_config)
        return config

    def get_otf_path(self, emission_wavelength_nm: int) -> Path | None:
        ws = self.get_channel_config(emission_wavelength_nm)
        if ws is None:
            raise ValueError(f"Channel '{emission_wavelength_nm}' is not configured")
        return ws.otf
