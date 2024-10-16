from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, Generic

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from mrc.mrc import Mrc

OptionalWavelengths = TypeVar("OptionalWavelengths", "Wavelengths", None)

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    UNSET = auto()
    PENDING = auto()
    RUNNING = auto()
    FAILED = auto()
    COMPLETE = auto()


@dataclass(frozen=True, slots=True)
class BoundMrc:
    array: NDArray[Any]
    mrc: Mrc


@dataclass(frozen=False, slots=True)
class ProcessingInfo:
    image_path: Path
    otf_path: Path
    config_path: Path
    output_path: Path
    log_path: Path
    wavelengths: Wavelengths
    kwargs: dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.UNSET


@dataclass(slots=True)
class ImageData:
    resolution: ImageResolution
    channels: tuple[ImageChannel, ...]


@dataclass(slots=True)
class ImageChannel(Generic[OptionalWavelengths]):
    wavelengths: OptionalWavelengths
    array: NDArray[Any] | None = None


@dataclass(slots=True, frozen=True)
class ImageResolution:
    x: float
    y: float
    z: float


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
