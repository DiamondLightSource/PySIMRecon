from __future__ import annotations

import logging
from pathlib import Path
from copy import deepcopy
import numpy as np
import mrc
from typing import TYPE_CHECKING, cast

from .dataclasses import ImageData, ImageChannel, ImageResolution, Wavelengths, BoundMrc
from ...exceptions import (
    PySimReconFileNotFoundError,
    PySimReconFileExistsError,
    InvalidValueError,
    PySimReconTypeError,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator, Collection
    from os import PathLike
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def read_dv(file_path: str | PathLike[str]) -> mrc.DVFile:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise PySimReconFileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    return mrc.DVFile(file_path)


def read_mrc_bound_array(file_path: str | PathLike[str]) -> BoundMrc:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise PySimReconFileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    bound_array = mrc.mrc.imread(str(file_path))
    return BoundMrc(bound_array, mrc=bound_array.Mrc)


def get_mrc_header_array(
    file_path: str | PathLike[str],
) -> np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]]:
    bound_mrc = read_mrc_bound_array(file_path)
    # This black magic is from the commented out bits of `makeHdrArray`.
    # Setting the memmap as a recarray, then `deepcopy`ing it allows the header
    # to be returned without requiring the large overall memmap to be kept open
    header_array = bound_mrc.mrc.hdr._array.view()
    return deepcopy(
        cast(
            np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]],
            header_array,
        )
    )


def get_dv_axis_order_from_header(dv: mrc.Mrc) -> str:
    sequence = dv.header.ImgSequence
    if sequence == 0:
        return "wtzyx"
    elif sequence == 1:
        return "tzwyx"
    elif sequence == 2:
        return "twzyx"
    raise InvalidValueError("DV header is invalid, ImgSequence must be 0, 1, or 2")


def get_dv_axis_sizes(dv: mrc.Mrc) -> dict[str, int]:
    header = dv.header
    return {
        "w": header.NumWaves,
        "t": header.NumTimes,
        "z": header.Num[2] // (header.NumWaves * header.NumTimes),
        "y": header.Num[1],
        "x": header.Num[0],
    }


def get_wavelengths_from_dv(dv: mrc.Mrc) -> Generator[Wavelengths, None, None]:

    def handle_rec_array(
        ext_floats: np.ndarray[Any, np.dtype[np.record]],
        wavelength_index: int,
        header_shape: tuple[int, ...],
    ) -> Generator[Wavelengths, None, None]:
        # Reshape header array to match the image ordering so we only have to
        # check each defined wavelength
        ext_floats = ext_floats.reshape(*header_shape)

        for c in range(header_shape[wavelength_index]):
            indexes = [0, 0, 0]
            indexes[wavelength_index] = c
            yield Wavelengths(
                # stored as a recarray of records, so getting the records need
                # to be indexed separately
                ext_floats[*indexes]["exWavelen"],
                ext_floats[*indexes]["emWavelen"],
            )

    def handle_float_array(
        ext_floats: NDArray[np.float32],
        wavelength_index: int,
        header_shape: tuple[int, ...],
    ) -> Generator[Wavelengths, None, None]:
        # If this is not a recarray, then it'll just be a big float32 array, so
        # an additional axis will be required for the metadata per plane
        ext_floats = ext_floats.reshape(*header_shape, -1)

        for c in range(header_shape[wavelength_index]):
            indexes = [0, 0, 0]
            indexes[wavelength_index] = c
            yield Wavelengths(  # indexed as [image frame index, float index]
                ext_floats[*indexes, 10],  # exWavelen index is 10
                ext_floats[*indexes, 11],  # emWavelen index is 11
            )

    ext_floats = dv.extFloats

    axis_order = get_dv_axis_order_from_header(dv)
    axis_sizes = get_dv_axis_sizes(dv)

    # Extended header is per frame, so don't include yx
    header_shape = tuple(axis_sizes[ax] for ax in axis_order[:3])

    wavelength_index = axis_order.index("w")

    if isinstance(ext_floats, np.recarray):
        return handle_rec_array(
            ext_floats,
            wavelength_index=wavelength_index,
            header_shape=header_shape,
        )
    elif isinstance(ext_floats, np.ndarray):
        return handle_float_array(
            ext_floats,
            wavelength_index=wavelength_index,
            header_shape=header_shape,
        )
    raise PySimReconTypeError(
        f"Extended header floats have an unexpected type {type(ext_floats)}"
    )


def write_dv(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    array: NDArray[Any],
    wavelengths: Collection[int],
    zoomfact: float,
    zzoom: int,
    overwrite: bool = False,
) -> Path:
    output_file = Path(output_file)

    logger.info(
        "Writing array to %s with wavelengths %s",
        output_file,
        ", ".join((str(w) for w in wavelengths)),
    )

    if output_file.is_file():
        if overwrite:
            logger.warning("Overwriting file %s", output_file)
            output_file.unlink()
        else:
            raise PySimReconFileExistsError(f"File {output_file} already exists")

    if len(wavelengths) != array.shape[-3]:
        raise InvalidValueError(
            "Length of wavelengths list must be equal to the number of channels in the array"
        )
    wave = [*wavelengths, 0, 0, 0, 0, 0][:5]
    # header_array = get_mrc_header_array(input_file)
    bound_mrc = read_mrc_bound_array(input_file)
    resolution = image_resolution_from_mrc(bound_mrc.mrc, warn_not_square=False)
    mrc.save(
        array,
        output_file,
        hdr=bound_mrc.mrc.hdr,
        metadata={
            "dx": resolution.x / zoomfact,
            "dy": resolution.y / zoomfact,
            "dz": resolution.z / zzoom,
            "wave": wave,
        },
    )
    logger.info(
        "%s saved",
        output_file,
    )
    return Path(output_file)


def get_image_data(
    file_path: str | PathLike[str],
) -> ImageData:
    file_path = Path(file_path)

    bound_mrc = read_mrc_bound_array(file_path)
    axis_order = get_dv_axis_order_from_header(bound_mrc.mrc)
    axis_sizes = get_dv_axis_sizes(bound_mrc.mrc)
    channel_index = axis_order.index("w")
    dv_shape = tuple(axis_sizes[ax] for ax in axis_order)
    resolution = image_resolution_from_mrc(bound_mrc.mrc, warn_not_square=True)

    # Essentially unsqueeze the axes to ensure the indexing is correct
    array = bound_mrc.array.reshape(dv_shape)

    channels: list[ImageChannel] = []
    wavelengths_tuple = tuple(get_wavelengths_from_dv(bound_mrc.mrc))
    num_channels = len(wavelengths_tuple)

    channel_dim_size = array.shape[channel_index]

    if channel_dim_size != num_channels:
        raise InvalidValueError(
            "The number of channels defined in the extended header "
            f"({num_channels}) don't match the size ({channel_dim_size}) of "
            f"the expected dimension ({channel_index})"
        )

    for c, wavelengths in enumerate(wavelengths_tuple):
        channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
        channel_slice[channel_index] = c
        channels.append(
            ImageChannel(
                wavelengths,
                array[*channel_slice].squeeze(),
            )
        )
    return ImageData(
        channels=tuple(channels),
        # Get resolution values from DV file (they get applied to TIFFs later)
        resolution=resolution,
    )


def image_resolution_from_mrc(
    mrc: mrc.mrc.Mrc, warn_not_square: bool = True
) -> ImageResolution:
    xyz_resolutions = mrc.header.d
    if warn_not_square and xyz_resolutions[0] != xyz_resolutions[1]:
        logger.warning("Pixels are not square in %s", mrc.path)
    return ImageResolution(
        x=xyz_resolutions[0],
        y=xyz_resolutions[1],
        z=xyz_resolutions[2],
    )
