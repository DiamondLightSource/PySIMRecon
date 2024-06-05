from __future__ import annotations
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pycudasirecon import reconstruct, make_otf, SIMReconstructor, ReconParams  # type: ignore[import-untyped]

from .files.config import read_input_config
from .prep import prepare_files
from .settings import SettingsManager

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
    from numpy.typing import NDArray


def create_otfs(
    channel_dict: NDArray[Any],
    psf_path: str | PathLike[str],
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
                os.unlink(otf_path)
            else:
                logging.info("Creating OTF for wavelength %f: %s", wavelength, otf_path)
            make_otf(
                psf=psf_array,  # says it accepts str but the functions inside accept arrays
                out_file=otf_path,
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


def reconstruct_wavelength(
    sim_array: NDArray[Any],
    otf_path: str | PathLike[str],
    output_directory: str | PathLike[str],
    output_stem: str,
    wavelength: int,
    xyres: float,
    zres: float,
    overwrite: bool = False,
    **params: Any,
) -> Path:
    output_directory = Path(output_directory)
    otf_path = Path(otf_path)
    if not otf_path.is_file():
        raise FileNotFoundError(f"OTF file {otf_path} does not exist")
    params = ReconParams(
        otf_file=str(otf_path),
        xyres=xyres,
        zres=zres,
        wavelength=wavelength,
        **params,
    )
    config_path = output_directory / f"{output_stem}_{wavelength}_cudasirecon.cfg"
    output_path = output_directory / f"{output_stem}_{wavelength}.mrc"
    save_params(
        config_path,
        params=params,
    )
    rec_array = run_recon(sim_array, config_path=config_path)
    SliceableMrc(data=rec_array).write(
        output_path,
        overwrite=overwrite,
    )
    return output_path


def run_recon(array: NDArray[Any], config_path: str | PathLike[str]) -> NDArray[Any]:
    return SIMReconstructor(array, config=config_path).get_result()

#TODO: Make this sensible and working
#TODO: Figure out when I'm converting PSFs vs looking fot OTFs
def reconstruct(
    image_path: str | PathLike[str],
    otf_paths: dict[int, str | PathLike[str]],
    output_directory: str | PathLike[str],
    overwrite: bool = False,
    **params: Any,
) -> None:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file {image_path} does not exist")


    intermediate_files = []

        for wavelength, wavelength_slice in wavelength_slices:
            try:
                int_wavelength = int(wavelength)
                otf_path = otf_paths[int_wavelength]
                intermediate_files.append(
                    reconstruct_wavelength(
                        sim_mrc[wavelength_slice],
                        otf_path=otf_path,
                        output_directory=output_directory,
                        output_stem=image_path.stem,
                        xyres=sim_mrc.voxel_size.x,
                        zres=sim_mrc.voxel_size.z,
                        overwrite=overwrite,
                        **params,
                    )
                )
            except Exception:
                logging.error("Failed to reconstruct wavelength %i", int_wavelength)


def save_params(config_path: str | PathLike[str], params: ReconParams) -> Path:
    with open(config_path, "w+") as f:
        f.write(params.to_config())


# def save_result(array: NDArray[Any], output_path: )


def run_reconstructions(
    output_directory: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    config_path: str | PathLike[str] | None = None,
    stitch_channels: bool = True,
    cleanup:bool=False,
    **kwargs,
) -> None:
    config = read_input_config(config_path)
    settings = SettingsManager(defaults_config=config.get("configs", "defaults"), wavelengths_config=config.get("configs", "wavelengths"))
    for sim_data_path in progress_wrapper(sim_data_paths, desc="SIM data files", unit="file"):
        prepare_files(sim_data_path, output_directory, settings=settings, cleanup=cleanup)
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
