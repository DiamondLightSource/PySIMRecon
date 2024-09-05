from __future__ import annotations

import logging
from pathlib import Path
from copy import deepcopy
import numpy as np
import mrc
from typing import TYPE_CHECKING, cast

from .dataclasses import ImageData, ImageChannel, ImageResolution, Wavelengths

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator, Collection
    from os import PathLike
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


def read_dv(file_path: str | PathLike[str]) -> mrc.DVFile:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    return mrc.DVFile(file_path)


def read_mrc_bound_array(file_path: str | PathLike[str]) -> NDArray[Any]:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    return mrc.mrc.imread(str(file_path))


def get_mrc_header_array(
    file_path: str | PathLike[str],
) -> np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]]:
    array = read_mrc_bound_array(file_path)
    # This black magic is from the commented out bits of `makeHdrArray`.
    # Setting the memmap as a recarray, then `deepcopy`ing it allows the header
    # to be returned without requiring the large overall memmap to be kept open
    header_array = array.Mrc.hdr._array.view()  # type: ignore[attr-defined]
    header_array.__class__ = np.recarray
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
    raise ValueError("DV header is invalid, ImgSequence must be 0, 1, or 2")


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
    ext_floats = dv.extFloats

    axis_order = get_dv_axis_order_from_header(dv)
    axis_sizes = get_dv_axis_sizes(dv)

    # Extended header is per frame, so don't include yx
    header_shape = tuple(axis_sizes[ax] for ax in axis_order[:3])
    ext_header = ext_floats.reshape(*header_shape, -1)

    wavelength_index = axis_order.index("w")

    for c in range(axis_sizes["w"]):
        indexes = [0, 0, 0]
        indexes[wavelength_index] = c
        yield Wavelengths(
            # indexed as [image frame index, float index]
            excitation_nm=ext_header[*indexes, 10],  # exWavelen index is 10
            emission_nm=ext_header[*indexes, 11],  # emWavelen index is 11
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
            raise FileExistsError(f"File {output_file} already exists")

    if len(wavelengths) != array.shape[-3]:
        raise ValueError(
            "Length of wavelengths list must be equal to the number of channels in the array"
        )
    wave = [*wavelengths, 0, 0, 0, 0, 0][:5]
    # header_array = get_mrc_header_array(input_file)
    input_data = read_mrc_bound_array(input_file)
    header = input_data.Mrc.hdr  # type: ignore
    mrc.save(
        array,
        output_file,
        hdr=header,
        metadata={
            "dx": header.d[2] / zoomfact,
            "dy": header.d[1] / zoomfact,
            "dz": header.d[0] / zzoom,
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

    array = read_mrc_bound_array(file_path)
    xyz_resolutions = array.Mrc.header.d
    if xyz_resolutions[0] != xyz_resolutions[1]:
        raise ValueError("DV file pixels are not square")

    axis_order = get_dv_axis_order_from_header(array.Mrc)
    axis_sizes = get_dv_axis_sizes(array.Mrc)
    channel_index = axis_order.index("w")
    dv_shape = tuple(axis_sizes[ax] for ax in axis_order)

    # Essentially unsqueeze the axes to ensure the indexing is correct
    array = array.reshape(dv_shape)

    channels: list[ImageChannel] = []
    wavelengths_tuple = tuple(get_wavelengths_from_dv(array.Mrc))
    num_channels = len(wavelengths_tuple)

    channel_dim_size = array.shape[channel_index]

    if channel_dim_size != num_channels:
        raise IndexError(
            "The number of channels defined in the extended header "
            f"({num_channels}) don't match the size ({channel_dim_size}) of "
            f"the expected dimension ({channel_index})"
        )

    for c, wavelengths in enumerate(wavelengths_tuple):
        channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
        channel_slice[channel_index] = c
        channels.append(
            ImageChannel(
                array[*channel_slice].squeeze(),
                wavelengths,
            )
        )
    return ImageData(
        channels=tuple(channels),
        # Get resolution values from DV file (they get applied to TIFFs later)
        resolution=ImageResolution(
            xy=xyz_resolutions[0], z=xyz_resolutions[2]  # Assumes square pixels
        ),
    )
