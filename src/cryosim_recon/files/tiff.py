from __future__ import annotations
from typing import TYPE_CHECKING, cast

import numpy as np
from ome_types import model
from ome_types.model.simple_types import PixelType, UnitsLength

from ..units import convert_to_unit


if TYPE_CHECKING:
    from numpy.typing import DTypeLike


__TIFF_TO_OME_UNITS_MAPPING: dict[int, UnitsLength] = {
    1: UnitsLength.REFERENCEFRAME,
    2: UnitsLength.INCH,
    3: UnitsLength.CENTIMETER,
    4: UnitsLength.MILLIMETER,
    5: UnitsLength.MICROMETER,
}

__DTYPE_DICT = {
    "int8": PixelType.INT8,
    "int16": PixelType.INT16,
    "int32": PixelType.INT32,
    "uint8": PixelType.UINT8,
    "uint16": PixelType.UINT16,
    "uint32": PixelType.UINT32,
    "float32": PixelType.FLOAT,
    "float64": PixelType.DOUBLE,
}


def handle_tiff_resolution(
    metadata: model.OME, resolution_unit: int
) -> tuple[float, float]:
    pixels = metadata.images[0].pixels

    x_size_info = (pixels.physical_size_x, pixels.physical_size_x_unit)
    y_size_info = (pixels.physical_size_y, pixels.physical_size_y_unit)

    if None in x_size_info or None in y_size_info:
        raise ValueError("Failed to interpret physical size from OME metadata")
    try:
        new_unit = __TIFF_TO_OME_UNITS_MAPPING[resolution_unit]
    except KeyError:
        raise ValueError(f"Unit {resolution_unit} is not a supported TIFF unit")
    return (
        1.0 / convert_to_unit(cast(float, x_size_info[0]), x_size_info[1], new_unit),
        1.0 / convert_to_unit(cast(float, y_size_info[0]), y_size_info[1], new_unit),
    )


def get_ome_pixel_type(dtype: DTypeLike) -> PixelType:
    try:
        return __DTYPE_DICT[np.dtype(dtype).name]
    except Exception:
        raise TypeError(
            f"{dtype} is unsupported data type. Supported dtypes are {', '.join(__DTYPE_DICT.keys())}.",
        )
