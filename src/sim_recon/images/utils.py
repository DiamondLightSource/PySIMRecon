from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

from ..exceptions import InvalidValueError, OutOfBoundsError

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def _calculate_shape_crop_slices(
    target_xy_shape: NDArray[np.integer[Any]],
    current_xy_shape: NDArray[np.integer[Any]],
    centre: NDArray[np.integer[Any]],
) -> tuple[slice, slice]:
    if np.any(target_xy_shape > current_xy_shape):
        raise OutOfBoundsError("Target crop shape exceeds at least one array dimension")

    from_centre = target_xy_shape / 2
    min_bounds = np.ceil(centre - from_centre).astype(np.uint16)
    max_bounds = np.ceil(centre + from_centre).astype(np.uint16)
    if np.any(min_bounds < 0) or np.any(max_bounds > current_xy_shape):
        raise OutOfBoundsError("Target crop extends beyond the bounds of the image")
    return slice(min_bounds[0], max_bounds[0]), slice(
        min_bounds[1], max_bounds[1]
    )  # x, y


def _calculate_decimal_crop_slices(
    crop: float,
    current_xy_shape: NDArray[np.integer[Any]],
    centre: NDArray[np.integer[Any]],
) -> tuple[slice, slice]:
    if crop <= 0 or crop >= 1:
        raise InvalidValueError("Argument 'crop' must be >= 0 and < 1")
    from_centre = (current_xy_shape * (1 - crop)) / 2
    min_bounds = np.ceil(centre - from_centre).astype(np.uint16)
    max_bounds = np.ceil(centre + from_centre).astype(np.uint16)
    if np.any(min_bounds < 0) or np.any(max_bounds > current_xy_shape):
        raise OutOfBoundsError("Target crop extends beyond the bounds of the image")
    return slice(min_bounds[0], max_bounds[0]), slice(
        min_bounds[1], max_bounds[1]
    )  # x, y


def apply_crop(
    array: NDArray[Any],
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
    xy_centre: tuple[int, int] | None = None,
) -> NDArray[Any]:

    assert array.ndim > 1, "Images must have > 1 dimension"

    array_slices = [slice(None)] * array.ndim

    current_xy_shape = np.asarray((array.shape[-1], array.shape[-2]), dtype=np.uint16)

    if xy_centre is None:
        # Crops more off the max than the min if not divisible by 2:
        centre = np.ceil((current_xy_shape - 1) / 2).astype(
            np.uint16
        )  # -1 to make it an index)
    else:
        centre = np.asarray(xy_centre, dtype=np.uint16)

    if xy_shape is not None:
        try:
            logger.info(
                "Cropping to X, Y: %i, %i centred on %i, %i",
                *xy_shape,
                *centre,
            )
            array_slices[-1:-3:-1] = _calculate_shape_crop_slices(
                np.asarray(xy_shape, dtype=np.uint16), current_xy_shape, centre
            )
        except OutOfBoundsError:
            raise OutOfBoundsError(
                f"Target shape {xy_shape} with centre {xy_centre} exceeds at least one array dimension"
            )
    elif crop != 0:
        try:
            logger.info("Cropping by %g%% centred on %i, %i", crop * 100, *centre)
            array_slices[-1:-3:-1] = _calculate_decimal_crop_slices(
                crop, current_xy_shape, centre
            )
        except OutOfBoundsError:
            raise OutOfBoundsError(
                f"Crop {crop:g}%% with centre {xy_centre} exceeds at least one array dimension"
            )
    elif xy_centre is not None:
        logger.warning("Argument 'xy_centre' ignored as no cropping was applied")
    else:
        logger.debug("No crop applied")

    return array[*array_slices]


def complex_to_interleaved_float(
    array: NDArray[np.complexfloating],
) -> NDArray[np.float32]:
    return array.view(np.float32)


def interleaved_float_to_complex(
    array: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    return array[:, :, 0::2] + 1j * array[:, :, 1::2]
