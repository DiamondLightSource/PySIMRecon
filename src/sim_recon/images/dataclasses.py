from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingInfo:
    image_path: Path
    otf_path: Path
    config_path: Path
    output_path: Path
    wavelengths: Wavelengths
    kwargs: dict[str, Any]


@dataclass(slots=True)
class ImageData:
    resolution: ImageResolution
    channels: tuple[ImageChannel, ...]


@dataclass(slots=True)
class ImageChannel:
    array: NDArray[Any] | None = None
    wavelengths: Wavelengths | None = None


@dataclass(slots=True, frozen=True)
class ImageResolution:
    x: float | None
    y: float | None
    z: float | None


@dataclass(slots=True, frozen=True)
class Wavelengths:
    excitation_nm: float | None = None
    emission_nm: float | None = None
    emission_nm_int: int | None = field(init=False)

    def __post_init__(self):
        if self.emission_nm is None:
            emission_nm_int = None
        else:
            emission_nm_int = int(round(self.emission_nm))
        object.__setattr__(self, "emission_nm_int", emission_nm_int)

    def __str__(self):
        return f"excitation: {self.excitation_nm}nm; emission: {self.emission_nm}nm"
