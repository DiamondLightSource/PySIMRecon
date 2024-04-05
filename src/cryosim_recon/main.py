from __future__ import annotations
import logging
import sys
from pathlib import Path
from subprocess import Popen, PIPE
from typing import TYPE_CHECKING, cast

import mrcfile
import numpy as np

from pycudadecon import make_otf

try:
    import tqdm

    progress_wrapper = tqdm
except ImportError:
    logging.info("tqdm not available, cannot monitor progress")

    def progress_wrapper(x, *args, **kwargs):
        return x


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray, DTypeLike


__OTF_FILENAME = "{0}_{wavelength}_otf.tiff"
__SPLIT_FILENAME = "{0}_{wavelength}.tiff"
__RECON_FILENAME = "{0}_{wavelength}_recon.tiff"


def read_dv(
    file_path: str | PathLike[str],
) -> tuple[
    tuple[float, float, float],
    dict[str, NDArray[Any]],
]:

    with mrcfile.open(file_path, mode="r", permissive=True, header_only=False) as f:
        voxel_dims = (
            float(f.voxel_size.x),
            float(f.voxel_size.y),
            float(f.voxel_size.z),
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


def create_otfs(
    channel_dict: NDArray[Any],
    psf_path: str | PathLike[str],
    dz: float,
    dx: float,
    overwrite: bool = False,
    **kwargs: Any,
) -> dict[float, Path]:
    psf_path = Path(psf_path)
    otfs_dict = {}
    for wavelength, psf_array in progress_wrapper(
        channel_dict.items(), desc=f"Creating OTFs from PSF: {psf_path}"
    ):
        otf_path = psf_path.parent / __OTF_FILENAME.format(
            psf_path.stem, wavelength=wavelength
        )
        if not otf_path.is_file() or overwrite:
            if overwrite:
                logging.warning(
                    "Overwriting OTF for wavelength %f: %s", wavelength, otf_path
                )
            else:
                logging.info("Creating OTF for wavelength %f: %s", wavelength, otf_path)
            make_otf(
                psf=psf_array,  # says it accepts str but the functions inside accept arrays
                outpath=otf_path,
                dzpsf=dz,
                dxpsf=dx,
                wavelength=wavelength,
                **kwargs,
            )
        else:
            logging.info(
                "OTF for wavelength %f already exists: %s", wavelength, otf_path
            )

        otfs_dict[wavelength] = otf_path
    return otfs_dict


def psf_to_otfs(
    psf_path: str | PathLike[str], overwrite: bool = False, **kwargs
) -> dict[float, Path]:
    psf_voxel_dims, psf_channel_dict = read_dv(psf_path)
    logging.info("Creating OTFs from psf: %s", psf_path)
    return create_otfs(
        channel_dict=psf_channel_dict,
        psf_path=psf_path,
        dz=psf_voxel_dims[2],
        dx=psf_voxel_dims[0],
        overwrite=overwrite,
        **kwargs,
    )


def create_recon_commands(
    image_path: str | PathLike[str],
    otf_path: str | PathLike[str],
    output_path: str | PathLike[str] = None,
    config: str | PathLike[str] = None,
    **kwargs,
) -> str:
    image_path = Path(image_path)
    command = ["condasirecon", str(image_path), str(output_path), str(otf_path)]
    if config is not None:
        command.extend(["-c", config])
    if kwargs:
        for key, value in kwargs.items():
            command.extend([f"--{key}", str(value)])
    return command


def run_command(command: list[str]) -> Popen:
    print(f"Running command: {' '.join(command)}")
    return Popen(command, stdin=None, stdout=PIPE, stderr=PIPE)


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


def run_reconstructions(
    output_directory: str | PathLike[str],
    otf_paths: dict[float, str | PathLike[str]],
    *sim_data_paths: str | PathLike[str],
    config: str | PathLike[str] | None = None,
    **kwargs,
) -> None:

    output_directory = Path(output_directory)
    for sim_path in progress_wrapper(
        sim_data_paths, desc="SIM data files", unit="file"
    ):
        logging.info("Splitting by emission wavelength: %s", sim_path)
        sim_path = Path(sim_path)
        voxel_dims, channel_image_dict = read_dv(sim_path)
        commands = set()
        for wavelength, image_stack in progress_wrapper(
            channel_image_dict.items(),
            desc=f"Splitting channels: {sim_path}",
            unit="channel",
        ):
            split_path = output_directory / __SPLIT_FILENAME.format(
                sim_path.stem, wavelength=wavelength
            )
            save_result(split_path, result=image_stack)
            commands.add(
                create_recon_commands(
                    image_path=split_path,
                    wavelength=wavelength,
                    output_path=sim_path.parent
                    / __RECON_FILENAME.format(sim_path.stem, wavelength),
                    otf_path=otf_paths[wavelength],
                    config=config,
                    **kwargs,
                )
            )
        processes: set[tuple[str, Popen]] = set()  # output file and process
        logging.info("Starting reconstructions: %s", sim_path)
        for command in tuple(commands):
            processes.add((command[3], run_command(command)))
            commands.discard(command)

        logging.info("Waiting for reconstructions to complete: %s", sim_path)
        for output_path, p in progress_wrapper(
            tuple(processes), desc=f"Reconstructions: {sim_path}", unit="channel"
        ):
            logging.debug("Waiting for: %s", output_path)
            p.wait()
            logging.info("Reconstruction complete: %s", output_path)
            processes.discard(p)
        logging.info("Reconstructions complete: %s", sim_path)


def run(
    output_directory: str | PathLike[str],
    psf_path: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
) -> None:
    psf_to_otfs(psf_path=psf_path, overwrite=True)
    logging.info("Starting reconstructions")
    run_reconstructions(output_directory, psf_path, *sim_data_paths)


if __name__ == "__main__":
    run(*sys.argv[1:])
