from __future__ import annotations
import logging
import sys
from pathlib import Path
from subprocess import Popen, PIPE
from typing import TYPE_CHECKING

from pycudadecon import make_otf

try:
    import tqdm

    progress_wrapper = tqdm
except ImportError:
    logging.info("tqdm not available, cannot monitor progress")

    def progress_wrapper(x, *args, **kwargs):
        return x


from .files import save, create_filename, read_dv


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray


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
        otf_path = psf_path.parent / create_filename(
            psf_path.stem, wavelength=wavelength, file_type="OTF", extension="TIFF"
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
    voxel_dims: tuple[float, float, float],
    psf_voxel_dims: tuple[float, float, float],
    wavelength: float,
    otf_path: str | PathLike[str],
    output_path: str | PathLike[str] = None,
    config: str | PathLike[str] = None,
    **kwargs,
) -> str:
    image_path = Path(image_path)
    command = [
        "condasirecon",
        str(image_path),
        str(output_path),
        str(otf_path),
        "--xyres",
        str(voxel_dims[0]),
        "--zres",
        str(voxel_dims[2]),
        "--wavelength",
        str(int(wavelength)),  # nm
        "--zresPSF",
        str(psf_voxel_dims[2]),
    ]
    if config is not None:
        command.extend(["-c", config])
    if kwargs:
        for key, value in kwargs.items():
            command.extend([f"--{key}", str(value)])
    return command


def run_command(command: list[str]) -> Popen:
    print(f"Running command: {' '.join(command)}")
    return Popen(command, stdin=None, stdout=PIPE, stderr=PIPE)


def run_reconstructions(
    output_directory: str | PathLike[str],
    otf_paths: dict[float, str | PathLike[str]],
    *sim_data_paths: str | PathLike[str],
    config: str | PathLike[str] | None = None,
    stitch_channels: bool = True,
    **kwargs,
) -> None:

    output_directory = Path(output_directory)
    for sim_path in progress_wrapper(
        sim_data_paths, desc="SIM data files", unit="file"
    ):
        logging.info("Splitting by emission wavelength: %s", sim_path)
        sim_path = Path(sim_path)
        dv_with_metadata = read_dv(sim_path)
        commands = set()
        output_paths = dict()
        for channel_with_metadata in progress_wrapper(
            dv_with_metadata.split_channels(),
            desc=f"Splitting channels: {sim_path}",
            unit="channel",
        ):
            split_path = output_directory / create_filename(
                sim_path.stem,
                wavelength=channel_with_metadata.wavelength[0],
                file_type="split",
                extension="TIFF",
            )
            save(split_path, result=channel_with_metadata.image)
            output_path = sim_path.parent / create_filename(
                sim_path.stem,
                wavelength=channel_with_metadata.wavelength[0],
                file_type="recon",
                extension="TIFF",
            )

            output_paths[channel_with_metadata.wavelength[0]] = output_path
            commands.add(
                create_recon_commands(
                    image_path=split_path,
                    wavelength=channel_with_metadata.wavelength[0],
                    output_path=output_path,
                    otf_path=otf_paths[channel_with_metadata.wavelength[0]],
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
        logging.info("Reconstructions complete: %s", sim_path)
        if stitch_channels:
            stitch_channels()


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
