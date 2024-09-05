from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def apply_crop(
    array: NDArray[Any],
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
) -> NDArray[Any]:
    assert array.ndim > 1, "Images must have > 1 dimension"
    array_slices = [slice(None)] * array.ndim
    if xy_shape is not None:
        logger.info("Cropping to X, Y: %i, %i", *xy_shape)
        target_yx_shape = np.asarray(xy_shape[::-1], dtype=np.uint16)
        current_yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
        crop_amount = current_yx_shape - target_yx_shape

        # Crops more off the max than the min if not divisible by 2:
        edge_crop_amount = crop_amount / 2
        min_bounds = np.floor(edge_crop_amount).astype(np.uint16)
        max_bounds = current_yx_shape - np.ceil(edge_crop_amount).astype(np.uint16)

        array_slices[-2] = slice(min_bounds[0], max_bounds[0])  # x
        array_slices[-1] = slice(min_bounds[1], max_bounds[1])  # y
    elif crop > 0 and crop <= 1:
        logger.info("Cropping by %g%%", crop * 100)
        yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
        min_bounds = np.round((yx_shape * crop) / 2).astype(np.uint16)
        max_bounds = yx_shape - min_bounds
        array_slices[-2] = slice(min_bounds[0], max_bounds[0])  # x
        array_slices[-1] = slice(min_bounds[1], max_bounds[1])  # y
    return array[*array_slices]


def complex_to_interleaved_float(
    array: NDArray[np.complexfloating],
) -> NDArray[np.float32]:
    return array.view(np.float32)


def interleaved_float_to_complex(
    array: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    return array[:, :, 0::2] + 1j * array[:, :, 1::2]
