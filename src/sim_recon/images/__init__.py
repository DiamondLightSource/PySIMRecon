from __future__ import annotations

import logging
import os
from pathlib import Path
import numpy as np
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .dv import get_image_data, write_mrc
from .tiff import read_tiff, write_tiff
from .utils import (
    apply_crop,
    complex_to_interleaved_float,
    interleaved_float_to_complex,
)
from ..exceptions import PySimReconTypeError

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
        if channel.array is not None:
            channel.array = apply_crop(channel.array, xy_shape=xy_shape, crop=crop)

            # TIFFs cannot handle complex values
            if np.iscomplexobj(channel.array):
                channel.array = complex_to_interleaved_float(channel.array)
    write_tiff(
        tiff_path,
        *image_data.channels,
        xy_pixel_size_microns=(image_data.resolution.x, image_data.resolution.y),
        overwrite=overwrite,
    )
    return Path(tiff_path)


def tiff_to_mrc(
    tiff_path: str | PathLike[str],
    mrc_path: str | PathLike[str],
    complex_output: bool = False,
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
    overwrite: bool = False,
) -> Path:
    array = read_tiff(tiff_path)
    is_floating = np.issubdtype(array.dtype, np.floating)
    if complex_output and is_floating:
        array = interleaved_float_to_complex(array)
    elif not is_floating:
        raise PySimReconTypeError(
            "Only floating type images can be converted to complex"
        )

    array = apply_crop(array, xy_shape=xy_shape, crop=crop)

    write_mrc(output_file=mrc_path, array=array, overwrite=overwrite)

    return Path(mrc_path)
