from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

import mrcfile
import numpy as np

from . import ImageWithMetadata


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray, DTypeLike


class HeaderIndexes(Enum):
    """
    DV file extended header indexes for float values

    Values Cockpit doesn't use have been commented out
    """

    # PhotosensorReading = 0
    ElapsedTime = 1
    StageCoordinateX = 2
    StageCoordinateY = 3
    StageCoordinateZ = 4
    # MinimumIntensity = 5
    # MaximumIntensity = 6
    # MeanIntensity = 7
    # ExposureTime = 8
    # NeutralDensity = 9
    ExcitationWavelength = 10
    EmissionWavelength = 11
    # IntensityScaling = 12
    # EnergyConversionFactor = 13


def read_dv(
    file_path: str | PathLike[str],
) -> ImageWithMetadata:

    with mrcfile.open(file_path, mode="r", permissive=True, header_only=False) as f:
        voxel_size = f.voxel_size
        extended_header = f.extended_header
        image_array = np.asarray(f.data)
    emission_wavelengths = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.EmissionWavelength
    )
    timestamps = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.ElapsedTime
    )
    excitation_wavelengths = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.ExcitationWavelength
    )
    emission_wavelengths = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.EmissionWavelength
    )
    xs = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.StageCoordinateX
    )
    ys = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.StageCoordinateY
    )
    zs = _get_eh_value_all_planes(
        extended_header, header_index=HeaderIndexes.StageCoordinateZ
    )
    return ImageWithMetadata(
        image=image_array,
        xs=xs,
        ys=ys,
        zs=zs,
        x_size=float(voxel_size.x),
        y_size=float(voxel_size.y),
        z_size=float(voxel_size.z),
        timestamps=timestamps,
        excitation_wavelengths=excitation_wavelengths,
        emission_wavelengths=emission_wavelengths,
    )


def _get_eh_value_all_planes(
    extended_header: NDArray[np.void], header_index: int, dtype: DTypeLike = np.float32
) -> np.generic:

    i = 0
    values = []
    try:
        for i in range(50000):  # something too big to be real
            values.append(
                _get_eh_value(
                    extended_header,
                    header_index=header_index,
                    plane_index=i,
                    dtype=dtype,
                )
            )
    except Exception:
        pass
    return tuple(values)


def _get_eh_value(
    extended_header: NDArray[np.void],
    header_index: int,
    plane_index: int,
    dtype: DTypeLike = np.float32,
) -> np.generic:
    """
    Interprets the dv format metadata from what the Cockpit developers know

    From https://microscope-cockpit.org/file-format

        The extended header has the following structure per plane

        8 32bit signed integers, often are all set to zero.

        Followed by 32 32bit floats. We only what the first 14 are:

        Float index | Meta data content
        ------------|----------------------------------------------
        0           | photosensor reading (typically in mV)
        1           | elapsed time (seconds since experiment began)
        2           | x stage coordinates
        3           | y stage coordinates
        4           | z stage coordinates
        5           | minimum intensity
        6           | maximum intensity
        7           | mean intensity
        8           | exposure time (seconds)
        9           | neutral density (fraction of 1 or percentage)
        10          | excitation wavelength
        11          | emission wavelength
        12          | intensity scaling (usually 1)
        13          | energy conversion factor (usually 1)

    """
    value_size_bytes = 4  # (32 / 8) first 8 are ints, rest are floats
    integer_bytes = 8 * value_size_bytes
    float_bytes = 32 * value_size_bytes
    plane_offset = (integer_bytes + float_bytes) * plane_index
    bytes_index = integer_bytes + plane_offset + header_index * value_size_bytes
    return np.frombuffer(
        extended_header,
        dtype=dtype,
        count=1,
        offset=bytes_index,
    )[0]
