from __future__ import annotations

import logging
import os
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .dv import get_image_data
from .tiff import write_tiff
from .utils import apply_crop, complex_to_interleaved_float

if TYPE_CHECKING:
    from collections.abc import Generator
    from os import PathLike

__all__ = ("dv_to_temporary_tiff", "dv_to_tiff")
logger = logging.getLogger(__name__)


@contextmanager
def dv_to_temporary_tiff(
    dv_path: str | PathLike[str],
    tiff_path: str | PathLike[str],
    delete: bool = False,
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
    overwrite: bool = False,
) -> Generator[Path, None, None]:
    try:
        yield dv_to_tiff(
            dv_path,
            tiff_path,
            xy_shape=xy_shape,
            crop=crop,
            overwrite=overwrite,
        )
    finally:
        if delete and tiff_path is not None:
            os.unlink(tiff_path)


def dv_to_tiff(
    dv_path: str | PathLike[str],
    tiff_path: str | PathLike[str],
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
    overwrite: bool = False,
) -> Path:
    image_data = get_image_data(dv_path)
    for channel in image_data.channels:
        channel.array = apply_crop(channel.array, xy_shape=xy_shape, crop=crop)

        # TIFFs cannot handle complex values
        if np.iscomplexobj(channel.array):
            channel.array = complex_to_interleaved_float(channel.array)
    write_tiff(
        tiff_path,
        *image_data.channels,
        pixel_size_microns=image_data.resolution.y,
        overwrite=overwrite,
    )
    return Path(tiff_path)
