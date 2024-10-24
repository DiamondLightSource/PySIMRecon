import pytest
from numpy.testing import assert_array_equal

import numpy as np

from sim_recon.images import utils
from sim_recon.exceptions import OutOfBoundsError, InvalidValueError


@pytest.mark.parametrize(
    "current_shape,target_shape,centre,expected_output,exception",
    [
        (
            np.asarray((10, 10), dtype=np.uint16),
            np.asarray((8, 8), dtype=np.uint16),
            np.asarray((4.5, 4.5)),
            (slice(1, 9), slice(1, 9)),
            None,
        ),
        (
            np.asarray((10, 10), dtype=np.uint16),
            np.asarray((6, 6), dtype=np.uint16),
            np.asarray((3.5, 6.5)),
            (slice(1, 7), slice(4, 10)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((3, 2), dtype=np.uint16),
            np.asarray((2, 2)),
            (slice(1, 4), slice(1, 3)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((3, 2), dtype=np.uint16),
            np.asarray((1, 4)),
            (slice(0, 3), slice(3, 5)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((6, 7), dtype=np.uint16),
            np.asarray((2, 2)),
            None,
            OutOfBoundsError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            np.asarray((2, 2), dtype=np.uint16),
            np.asarray((6, 2)),
            None,
            OutOfBoundsError,
        ),
    ],
    ids=[
        "10x no offset",
        "10x offset",
        "5x no offset",
        "5x offset",
        "shape too large",
        "shifted out of bounds",
    ],
)
def test_calculate_2d_shape_crop_slices(
    current_shape, target_shape, centre, expected_output, exception
):
    kwargs = {
        "target_shape": target_shape,
        "current_shape": current_shape,
        "centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils._calculate_2d_shape_crop_slices(**kwargs)
    else:
        assert utils._calculate_2d_shape_crop_slices(**kwargs) == expected_output


@pytest.mark.parametrize(
    "current_shape,crop,centre,expected_output,exception",
    [
        (
            np.asarray((5, 5), dtype=np.uint16),
            0.4,  # 2 / 5
            np.asarray((2, 2)),
            (slice(1, 4), slice(1, 4)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            0.4,
            np.asarray((1, 3)),
            (slice(0, 3), slice(2, 5)),
            None,
        ),
        (
            np.asarray((10, 10), dtype=np.uint16),
            0.2,
            np.asarray((4.5, 4.5)),
            (slice(1, 9), slice(1, 9)),
            None,
        ),
        (
            np.asarray((10, 10), dtype=np.uint16),
            0.4,
            np.asarray((3.5, 6.5)),
            (slice(1, 7), slice(4, 10)),
            None,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            2,
            np.asarray((2, 2)),
            None,
            InvalidValueError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            -1,
            np.asarray((2, 2)),
            None,
            InvalidValueError,
        ),
        (
            np.asarray((5, 5), dtype=np.uint16),
            0.1,
            np.asarray((6, 2)),
            None,
            OutOfBoundsError,
        ),
    ],
    ids=[
        "5x no offset",
        "5x offset",
        "10x no offset",
        "10x offset",
        "crop >= 1",
        "crop <= 0",
        "shifted out of bounds",
    ],
)
def test_calculate_2d_decimal_crop_slices(
    current_shape, crop, centre, expected_output, exception
):
    kwargs = {
        "crop": crop,
        "current_shape": current_shape,
        "centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils._calculate_2d_decimal_crop_slices(**kwargs)
    else:
        assert utils._calculate_2d_decimal_crop_slices(**kwargs) == expected_output


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
            0,
            None,
            np.arange(2, 4).reshape(1, -1) * np.arange(2, 5).reshape(-1, 1),
            None,
        ),
        (
            np.stack(
                [np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1)] * 3,
                axis=0,
            ),
            np.asarray((3, 2), dtype=np.uint16),
            0,
            None,
            np.stack(
                [np.arange(2, 4).reshape(1, -1) * np.arange(2, 5).reshape(-1, 1)] * 3,
                axis=0,
            ),
            None,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((3, 2), dtype=np.uint16),
            0,
            np.asarray((2, 3)),
            np.arange(3, 5).reshape(1, -1) * np.arange(2, 5).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((7, 6), dtype=np.uint16),
            0,
            None,
            None,
            OutOfBoundsError,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((2, 2), dtype=np.uint16),
            0,
            np.asarray((3, 7)),
            None,
            OutOfBoundsError,
        ),
        (
            np.arange(1, 6).reshape(1, -1) * np.arange(1, 6).reshape(-1, 1),
            np.asarray((3, 2)),
            0.7,
            None,
            np.arange(2, 4).reshape(1, -1) * np.arange(2, 5).reshape(-1, 1),
            None,
        ),
        (
            np.arange(1, 11).reshape(1, -1) * np.arange(1, 11).reshape(-1, 1),
            None,
            0.2,
            np.asarray((4.5, 4.5)),
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
            np.asarray((3.5, 6.5)),
            np.arange(5, 11).reshape(1, -1) * np.arange(2, 8).reshape(-1, 1),
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
            np.asarray((3, 1)),
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
        "shape": target_shape,
        "crop": crop,
        "centre": centre,
    }
    if exception is not None:
        with pytest.raises(exception):
            utils.apply_crop(**kwargs)
    else:
        result = utils.apply_crop(**kwargs)
        result_2d_shape = np.asarray(result.shape[-2::])
        if target_shape is not None:
            assert_array_equal(
                result_2d_shape, target_shape, err_msg="expected shape does not match"
            )
        elif crop != 0:
            expected_shape = np.round(
                np.asarray(array.shape[-2::]) * (1.0 - crop)
            ).astype(np.uint16)
            assert_array_equal(
                result_2d_shape, expected_shape, err_msg="expected shape does not match"
            )
        assert_array_equal(
            result, expected_output, err_msg="result doesn't match expected output"
        )
