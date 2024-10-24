from __future__ import annotations

import logging
import numpy as np
from typing import TYPE_CHECKING

from ..exceptions import InvalidValueError, OutOfBoundsError

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def __get_bounds_from_centre(
    centre: NDArray[np.floating], from_centre: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    # np.floor to crop more off the min than the max if not divisible by 2
    min_bounds = np.floor(centre - from_centre)
    max_bounds = np.floor(
        centre + from_centre + 1  # slices are not inclusive of top bound
    )
    if np.any(min_bounds < 0) or np.any(max_bounds < 0):
        raise OutOfBoundsError("Target crop extends beyond the bounds of the image")
    return min_bounds, max_bounds


def _calculate_2d_shape_crop_slices(
    target_shape: NDArray[np.integer[Any]],
    current_shape: NDArray[np.integer[Any]],
    centre: NDArray[np.floating[Any]],
) -> tuple[slice, slice]:
    if np.any(target_shape > current_shape):
        raise OutOfBoundsError("Target crop shape exceeds at least one array dimension")
    from_centre = (target_shape - 1) / 2
    min_bounds, max_bounds = __get_bounds_from_centre(centre, from_centre)
    if np.any(max_bounds > current_shape):
        raise OutOfBoundsError("Target crop extends beyond the bounds of the image")
    return (
        slice(int(min_bounds[0]), int(max_bounds[0])),
        slice(int(min_bounds[1]), int(max_bounds[1])),
    )


def _calculate_2d_decimal_crop_slices(
    crop: float,
    current_shape: NDArray[np.integer[Any]],
    centre: NDArray[np.floating[Any]],
) -> tuple[slice, slice]:
    if crop <= 0 or crop >= 1:
        raise InvalidValueError("Argument 'crop' must be > 0 and < 1")
    from_centre = (current_shape * (1 - crop) - 1) / 2
    min_bounds, max_bounds = __get_bounds_from_centre(centre, from_centre)
    if np.any(max_bounds > current_shape):
        raise OutOfBoundsError("Target crop extends beyond the bounds of the image")
    return (
        slice(int(min_bounds[0]), int(max_bounds[0])),
        slice(int(min_bounds[1]), int(max_bounds[1])),
    )


def apply_crop(
    array: NDArray[Any],
    shape: tuple[int, int] | None = None,
    crop: float = 0,
    centre: tuple[float, float] | None = None,
) -> NDArray[Any]:

    assert array.ndim > 1, "Images must have > 1 dimension"

    array_slices = [slice(None)] * array.ndim

    current_2d_shape = np.asarray((array.shape[-2], array.shape[-1]), dtype=np.uint16)

    if centre is None:
        # -1 to make it an index
        cen = (current_2d_shape - 1) / 2
    else:
        cen = np.asarray(centre)

    if shape is not None:
        try:
            logger.info(
                "Cropping to X, Y: %i, %i centred on X, Y: %i, %i",
                shape[1],
                shape[0],
                cen[1],
                cen[0],
            )
            array_slices[-2::] = _calculate_2d_shape_crop_slices(
                np.asarray(shape, dtype=np.uint16), current_2d_shape, cen
            )
        except OutOfBoundsError:
            raise OutOfBoundsError(
                "Target shape X, Y: %i, %i with centre X, Y: %i, %i exceeds at least one array dimension"
                % (shape[1], shape[0], cen[1], cen[0])
            )
    elif crop != 0:
        try:
            logger.info(
                "Cropping by %g%% centred on X, Y: %i, %i", crop * 100, cen[1], cen[0]
            )
            array_slices[-2::] = _calculate_2d_decimal_crop_slices(
                crop, current_2d_shape, cen
            )
        except OutOfBoundsError:
            raise OutOfBoundsError(
                "Crop %g%% with centre X, Y: %i, %i exceeds at least one array dimension"
                % (crop, cen[1], cen[0])
            )
    elif centre is not None:
        logger.warning("Argument 'centre' ignored as no cropping was applied")
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
