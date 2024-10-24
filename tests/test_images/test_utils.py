import pytest
from numpy.testing import assert_array_equal

import numpy as np

from sim_recon.images import utils
from sim_recon.exceptions import OutOfBoundsError, InvalidValueError


@pytest.mark.parametrize(
    "current_shape,target_shape,centre,expected_output,exception",
    [
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((3, 2), dtype=np.uint16),
            np.asarray((2, 2), dtype=np.uint16),
            (slice(1, 4), slice(1, 3)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((3, 2), dtype=np.uint16),
            np.asarray((1, 4), dtype=np.uint16),
            (slice(0, 3), slice(3, 5)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((6, 7), dtype=np.uint16),
            np.asarray((2, 2), dtype=np.uint16),
            None,
            OutOfBoundsError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((2, 2), dtype=np.uint16),
            np.asarray((6, 2), dtype=np.uint16),
            None,
            OutOfBoundsError,
        ),
    ],
    ids=["no offset", "offset", "shape too large", "shifted out of bounds"],
)
def test_calculate_shape_crop_slices(
    current_shape, target_shape, centre, expected_output, exception
):
    kwargs = {
        "target_xy_shape": target_shape,
        "current_xy_shape": current_shape,
        "centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils._calculate_shape_crop_slices(**kwargs)
    else:
        assert utils._calculate_shape_crop_slices(**kwargs) == expected_output


@pytest.mark.parametrize(
    "current_shape,crop,centre,expected_output,exception",
    [
        (
            np.asarray((10, 10), dtype=np.uint16),
            0.2,
            np.asarray((5, 5), dtype=np.uint16),
            (slice(1, 9), slice(1, 9)),
            None,
        ),
        (
            np.asarray((10, 10), dtype=np.uint16),
            0.4,
            np.asarray((4, 7), dtype=np.uint16),
            (slice(1, 7), slice(4, 10)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            2,
            np.asarray((2, 2), dtype=np.uint16),
            None,
            InvalidValueError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            -1,
            np.asarray((2, 2), dtype=np.uint16),
            None,
            InvalidValueError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            0.1,
            np.asarray((6, 2), dtype=np.uint16),
            None,
            OutOfBoundsError,
        ),
    ],
    ids=["no offset", "offset", "crop >= 1", "crop <= 0", "shifted out of bounds"],
)
def test_calculate_decimal_crop_slices(
    current_shape, crop, centre, expected_output, exception
):
    kwargs = {
        "crop": crop,
        "current_xy_shape": current_shape,
        "centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils._calculate_decimal_crop_slices(**kwargs)
    else:
        assert utils._calculate_decimal_crop_slices(**kwargs) == expected_output


@pytest.mark.parametrize(
    "array,target_shape,crop,centre,expected_output,exception",
    [
        (
            np.arange(1, 3).reshape(1, -1) * np.arange(1, 3).reshape(-1, 1),
            None,
            0,
            None,
            np.arange(1, 3).reshape(1, -1) * np.arange(1, 3).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((3, 2), dtype=np.uint16),
            None,
            None,
            np.arange(2, 5).reshape(1, -1) * np.arange(2, 4).reshape(-1, 1),
            None,
        ),
        (
            np.stack(
                [np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1)] * 3,
                axis=0,
            ),
            np.asarray((3, 2), dtype=np.uint16),
            None,
            None,
            np.stack(
                [np.arange(2, 5).reshape(1, -1) * np.arange(2, 4).reshape(-1, 1)] * 3,
                axis=0,
            ),
            None,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((3, 2), dtype=np.uint16),
            None,
            np.asarray((2, 4), dtype=np.uint16),
            np.arange(2, 5).reshape(1, -1) * np.arange(4, 6).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((6, 7), dtype=np.uint16),
            None,
            None,
            None,
            OutOfBoundsError,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((2, 2), dtype=np.uint16),
            None,
            np.asarray((7, 3), dtype=np.uint16),
            None,
            OutOfBoundsError,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((3, 2), dtype=np.uint16),
            0.7,
            None,
            np.arange(2, 5).reshape(1, -1) * np.arange(2, 4).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 11).reshape(1, -1) * np.arange(1, 11).reshape(-1, 1),
            None,
            0.2,
            np.asarray((5, 5), dtype=np.uint16),
            np.arange(2, 10).reshape(1, -1) * np.arange(2, 10).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 11).reshape(1, -1) * np.arange(1, 11).reshape(-1, 1),
            None,
            0.2,
            None,
            np.arange(2, 10).reshape(1, -1) * np.arange(2, 10).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 11).reshape(1, -1) * np.arange(1, 11).reshape(-1, 1),
            None,
            0.4,
            np.asarray((3, 6), dtype=np.uint16),
            np.arange(1, 7).reshape(1, -1) * np.arange(4, 10).reshape(-1, 1),
            None,
        ),
        (
            np.arange(2).reshape(1, -1) * np.arange(2).reshape(-1, 1),
            None,
            2,
            None,
            None,
            InvalidValueError,
        ),
        (
            np.arange(2).reshape(1, -1) * np.arange(2).reshape(-1, 1),
            None,
            -1,
            None,
            None,
            InvalidValueError,
        ),
        (
            np.arange(2).reshape(1, -1) * np.arange(2).reshape(-1, 1),
            None,
            0.1,
            np.asarray([3, 1], dtype=np.uint16),
            None,
            OutOfBoundsError,
        ),
    ],
    ids=[
        "unchanged",
        "shape no offset",
        "shape stack",
        "shape offset",
        "shape too large",
        "shifted out of bounds",
        "shape overrides crop",
        "crop no offset",
        "crop none offset",
        "crop offset",
        "crop >= 1",
        "crop < 0",
        "crop shifted out of bounds",
    ],
)
def test_crop(array, target_shape, crop, centre, expected_output, exception):
    kwargs = {
        "array": array,
        "xy_shape": target_shape,
        "crop": crop,
        "xy_centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils.apply_crop(**kwargs)
    else:
        result = utils.apply_crop(**kwargs)
        result_xy_shape = np.asarray(result.shape[-1:-3:-1])
        if target_shape is not None:
            assert_array_equal(
                result_xy_shape, target_shape, err_msg="expected shape does not match"
            )
        elif crop != 0:
            expected_shape = np.round(
                np.asarray(array.shape[-1:-3:-1]) * (1.0 - crop)
            ).astype(np.uint16)
            assert_array_equal(
                result_xy_shape, expected_shape, err_msg="expected shape does not match"
            )
        assert_array_equal(
            result, expected_output, err_msg="result doesn't match expected output"
        )


def test_complex_to_interleaved_float(): ...


def test_interleaved_float_to_complex(): ...
