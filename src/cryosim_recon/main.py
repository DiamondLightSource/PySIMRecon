from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mrcfile
import numpy as np
from pycudadecon import RLContext, TemporaryOTF, rl_decon


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray


def read_dv(
    file_path: str | PathLike[str],
) -> tuple[
    tuple[float, float, float],
    dict[str, NDArray[Any]],
]:

    with mrcfile.open(file_path, permissive=True) as f:
        voxel_dims = (
            f.voxel_size.x,
            f.voxel_size.y,
            f.voxel_size.z,
        )
        extended_header = f.extended_header
        image_array = np.asarray(f.data)
    return voxel_dims, split_dv_channels(image_array, extended_header=extended_header)


def split_dv_channels(
    image_stack_array: NDArray[Any], extended_header: NDArray[np.void]
) -> dict[str, NDArray[Any]]:

    def get_value(
        extended_header: NDArray[np.void],
        header_index: int,
        plane_index: int,
        dtype: DTypeLike,
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

    channel_images = {}
    current_emission_wavelength = -1
    channel_start = 0
    for plane_index in range(len(image_stack_array)):
        emission_wavelength = cast(
            float,
            get_value(
                extended_header,
                header_index=11,
                plane_index=plane_index,
                dtype=np.float32,
            ),
        )
        if emission_wavelength != current_emission_wavelength:
            if current_emission_wavelength != -1:
                if current_emission_wavelength in channel_images:
                    raise KeyError(
                        f"Emission wavelength {current_emission_wavelength} already exists"
                    )
                channel_images[current_emission_wavelength] = image_stack_array[
                    channel_start:plane_index, :, :
                ]
            channel_start = plane_index
            current_emission_wavelength = emission_wavelength
        # Add final channel
        channel_images[current_emission_wavelength] = image_stack_array[
            channel_start : plane_index + 1, :, :
        ]
    return channel_images



def save_result(
    output_path: str | PathLike[str],
    result: NDArray[Any] | tuple[NDArray[Any], NDArray[Any]],
) -> None:
    output_path = Path(output_path)
    if not output_path.parent.is_dir():
        raise IOError(
            f"Cannot save to directory {output_path.parent} that does not exist"
        )
    if isinstance(result, tuple):
        deskewed_path = (
            output_path.parent / f"{output_path.stem}_deskewed{output_path.suffix}"
        )
        mrcfile.write(deskewed_path, result[1])
        result = result[0]

    mrcfile.write(output_path, result)


def run(
    output_directory: str | PathLike[str],
    psf_path: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
) -> None:
    psf_voxel_dims, psf_channel_dict = read_dv(psf_path)
    output_directory = Path(output_directory)
    for sim_path in sim_data_paths:
        sim_path = Path(sim_path)
        voxel_dims, channel_image_dict = read_dv(sim_path)
        for wavelength, image_stack in channel_image_dict.items():
            save_result(
                output_path=output_directory / f"{sim_path.stem}_{wavelength}.mrc",
                result=run_recon(
                    image_stack,
                    psf_array=psf_channel_dict[wavelength],
                    **{
                        "dzdata": voxel_dims[2],
                        "dxdata": voxel_dims[0],
                        "dzpsf": psf_voxel_dims[2],
                        "dxpsf": psf_voxel_dims[0],
                    },
                ),
            )


if __name__ == "__main__":
    run(*sys.argv[1:])
