from __future__ import annotations
import logging
import subprocess
import multiprocessing
import os
import traceback
from os.path import abspath
from shutil import copyfile
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pycudasirecon.sim_reconstructor import SIMReconstructor, lib  # type: ignore[import-untyped]

from .files.utils import redirect_output_to, create_output_path, combine_text_files
from .files.config import create_wavelength_config
from .images import get_image_data
from .images.dv import write_dv
from .images.tiff import read_tiff, write_tiff, get_combined_array_from_tiffs
from .images.dataclasses import ImageChannel, Wavelengths, ProcessingInfo
from .settings import ConfigManager
from .settings.formatting import formatters_to_default_value_kwargs, RECON_FORMATTERS
from .progress import get_progress_wrapper, get_logging_redirect

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from multiprocessing.pool import AsyncResult
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class ReconstructionException(Exception):
    pass


def _recon_get_result(
    reconstructor: SIMReconstructor, output_shape: tuple[int, int, int]
) -> NDArray[np.float32]:
    """
    Equivalent of `SIMReconstructor.get_result()`

    `SIMReconstructor.get_recon_params()` returns some invalid values (0s for `z_zoom` and `zoomfact`) which breaks
    `lib.SR_getResult`. To work around this, define the expected output shape manually.
    """
    _result = np.empty(output_shape, np.float32)
    lib.SR_getResult(reconstructor._ptr, _result)
    return _result


def subprocess_recon(
    sim_path: Path, otf_path: Path, config_path: Path
) -> NDArray[np.float32]:
    """Useful to bypass the pycudasirecon library, if necessary"""
    p = subprocess.run(
        [
            "cudasirecon",
            str(sim_path.parent),
            sim_path.name,
            str(otf_path),
            "-c",
            str(config_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    expected_path = sim_path.parent / "GPUsirecon" / f"{sim_path.stem}_proc.tif"
    if str(expected_path) in p.stdout:
        sim_path.with_suffix(".log").write_text(p.stdout)
        array = read_tiff(expected_path)
        expected_path.unlink()
        return array
    raise ReconstructionException(f"No reconstruction file found at {expected_path}")


def reconstruct(
    array: NDArray[Any],
    config_path: str | PathLike[str],
    zoomfact: float,
    zzoom: int,
    ndirs: int,
    nphases: int,
) -> NDArray[np.float32]:
    """Does not accept memmaps"""
    try:
        reconstructor = SIMReconstructor(array, config=str(abspath(config_path)))
        z, y, x = array.shape
        z_div = ndirs * nphases
        if z % z_div != 0:
            raise ReconstructionException(
                f"Z size {z} is not divisible by the number of phases and angles ({nphases} * {ndirs})"
            )
        z = (z // z_div) * zzoom
        x = int(round(x * zoomfact))
        y = int(round(y * zoomfact))

        return _recon_get_result(reconstructor, output_shape=(z, y, x))
        # return reconstructor.get_result()
    except Exception:
        # Unlikely to ever hit this as errors from the C++ just kill the process
        logger.error(
            "Exception raised during reconstruction with config %s",
            config_path,
            exc_info=True,
        )
        raise ReconstructionException(
            f"Exception raised during reconstruction with config {config_path}"
        )


def reconstruct_from_processing_info(processing_info: ProcessingInfo) -> Path:
    logger.info(
        "Starting reconstruction of %s with %s to be saved as %s",
        processing_info.image_path,
        processing_info.config_path,
        processing_info.output_path,
    )
    # Get parameters from kwargs for scaling outputs
    zoomfact = float(processing_info.kwargs["zoomfact"])  # stored as Decimal
    zzoom = processing_info.kwargs["zzoom"]
    ndirs = processing_info.kwargs["ndirs"]
    nphases = processing_info.kwargs["nphases"]

    data = read_tiff(processing_info.image_path)  # Cannot use a memmap here!

    with redirect_output_to(processing_info.log_path):
        print(
            "\n".join(
                (
                    f"Channel {processing_info.wavelengths.emission_nm_int} ({processing_info.wavelengths})",
                    "Config used:",
                    processing_info.config_path.read_text().strip(),
                    "-" * 80,
                    "The text below is the output from cudasirecon",
                )
            )
        )
        rec_array = reconstruct(
            data, processing_info.config_path, zoomfact, zzoom, ndirs, nphases
        )

    # rec_array = subprocess_recon(
    #     processing_info.image_path,
    #     processing_info.otf_path,
    #     processing_info.config_path,
    # )
    logger.info("Reconstructed %s", processing_info.image_path)
    recon_pixel_size = float(processing_info.kwargs["xyres"]) / zoomfact
    write_tiff(
        processing_info.output_path,
        ImageChannel(processing_info.wavelengths, rec_array),
        xy_pixel_size_microns=(recon_pixel_size, recon_pixel_size),
        overwrite=True,
    )
    logger.debug(
        "Reconstruction of %s saved in %s",
        processing_info.image_path,
        processing_info.output_path,
    )
    return Path(processing_info.output_path)


def run_reconstructions(
    conf: ConfigManager,
    *sim_data_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None,
    overwrite: bool = False,
    cleanup: bool = False,
    stitch_channels: bool = True,
    parallel_process: bool = False,
    **config_kwargs: Any,
) -> None:

    logging_redirect = get_logging_redirect()
    progress_wrapper = get_progress_wrapper()

    # `maxtasksperchild=1` is necessary to ensure the child process is cleaned
    # up between tasks, as the cudasirecon process doesn't fully release memory
    # afterwards
    with (
        multiprocessing.Pool(
            processes=1 + int(parallel_process),  # 2 processes max
            maxtasksperchild=1,
        ) as pool,
        logging_redirect(),
    ):
        for sim_data_path in progress_wrapper(
            sim_data_paths, desc="SIM data files", unit="file"
        ):
            try:
                sim_data_path = Path(sim_data_path)
                file_output_directory = (
                    sim_data_path.parent
                    if output_directory is None
                    else Path(output_directory)
                )
                if not sim_data_path.is_file():
                    raise FileNotFoundError(
                        f"Image file {sim_data_path} does not exist"
                    )

                processing_directory: str | Path
                with TemporaryDirectory(
                    prefix="proc_",
                    suffix=f"_{sim_data_path.stem}",
                    dir=file_output_directory,
                    delete=cleanup,
                ) as processing_directory:
                    processing_directory = Path(processing_directory)

                    # These processing files are cleaned up by TemporaryDirectory
                    # As single-wavelength files will be used directly and we don't
                    # want to delete real input files!
                    processing_info_dict = _prepare_files(
                        sim_data_path,
                        processing_directory,
                        conf=conf,
                        **config_kwargs,
                    )
                    if not processing_info_dict:
                        logger.warning("Skipping %s (unable to process)", sim_data_path)
                        continue

                    async_results: list[AsyncResult] = []
                    zoom_factors: list[tuple[float, int]] = []
                    for wavelength, processing_info in processing_info_dict.items():
                        async_results.append(
                            pool.apply_async(
                                reconstruct_from_processing_info,
                                args=(processing_info,),
                                error_callback=lambda e: logger.error(
                                    # exc_info doesn't work with this
                                    "Error occurred during reconstruction of %s channel %i: %s",
                                    sim_data_path,
                                    wavelength,
                                    "".join(traceback.format_exception(e)),
                                ),
                            )
                        )

                        # Get zoom factors to for checking if the output can be stitched
                        zoom_factors.append(
                            (
                                processing_info.kwargs["zoomfact"],
                                processing_info.kwargs["zzoom"],
                            )
                        )

                    # Wait for async processes to finish (otherwise there won't be files to stitch!)
                    for r in progress_wrapper(
                        async_results,
                        unit="wavelengths",
                        leave=False,
                    ):
                        r.wait()

                    # Check if images can be stitched
                    if stitch_channels and zoom_factors.count(zoom_factors[0]) != len(
                        zoom_factors
                    ):
                        logger.warning(
                            "Unable to stitch files due to mismatched zoom factors between wavelengths"
                        )
                        stitch_channels = False

                    if stitch_channels:
                        # Stitch channels (if requested and possible)
                        dv_path = create_output_path(
                            sim_data_path,
                            output_type="recon",
                            suffix=".dv",
                            output_directory=file_output_directory,
                            ensure_unique=not overwrite,
                        )

                        combine_text_files(
                            dv_path.with_suffix(".log"),
                            *(pi.log_path for pi in processing_info_dict.values()),
                            header="Reconstruction log",
                        )

                        write_dv(
                            sim_data_path,
                            dv_path,
                            get_combined_array_from_tiffs(
                                *(
                                    pi.output_path
                                    for pi in processing_info_dict.values()
                                )
                            ),
                            wavelengths=tuple(processing_info_dict.keys()),
                            zoomfact=float(zoom_factors[0][0]),
                            zzoom=zoom_factors[0][1],
                            overwrite=overwrite,
                        )
                    else:
                        for (
                            wavelength,
                            processing_info,
                        ) in processing_info_dict.items():
                            dv_path = create_output_path(
                                sim_data_path,
                                output_type="recon",
                                suffix=".dv",
                                output_directory=file_output_directory,
                                wavelength=wavelength,
                                ensure_unique=not overwrite,
                            )

                            combine_text_files(
                                dv_path.with_suffix(".log"),
                                *(pi.log_path for pi in processing_info_dict.values()),
                                header="Reconstruction log",
                            )

                            write_dv(
                                sim_data_path,
                                dv_path,
                                get_combined_array_from_tiffs(
                                    processing_info.output_path
                                ),
                                wavelengths=(wavelength,),
                                zoomfact=float(processing_info.kwargs["zoomfact"]),
                                zzoom=processing_info.kwargs["zzoom"],
                            )

                    if cleanup:
                        for processing_info in processing_info_dict.values():
                            try:
                                if processing_info.output_path.is_file():
                                    logger.debug(
                                        "Removing %s", processing_info.output_path
                                    )
                                    os.remove(processing_info.output_path)
                            except Exception:
                                logger.error(
                                    "Failed to remove %s",
                                    processing_info.output_path,
                                    exc_info=True,
                                )

            except Exception:
                logger.error("Error occurred for %s", sim_data_path, exc_info=True)


def _prepare_config_kwargs(
    conf: ConfigManager,
    emission_wavelength: int,
    otf_path: str | PathLike[str],
    **config_kwargs: Any,
) -> dict[str, Any]:
    # Get the default values (for settings with default values)
    kwargs = formatters_to_default_value_kwargs(RECON_FORMATTERS)

    # Use the configured per-wavelength settings
    kwargs.update(conf.get_reconstruction_config(emission_wavelength))

    # config_kwargs override those any config defaults set
    kwargs.update(config_kwargs)

    # Set final variables:
    kwargs["wavelength"] = emission_wavelength
    # Add otf_file that is expected by ReconParams
    # Needs to be absolute because we don't know where this might be run from
    kwargs["otf_file"] = str(Path(otf_path).absolute())

    return kwargs


def create_processing_info(
    file_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    wavelengths: Wavelengths,
    conf: ConfigManager,
    **config_kwargs: Any,
) -> ProcessingInfo:
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot create processing info: file {file_path} does not exist"
        )
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)

    if wavelengths.emission_nm_int is None:
        raise AttributeError("emission_nm_int", wavelengths)
    otf_path = conf.get_otf_path(wavelengths.emission_nm_int)

    if otf_path is None:
        raise ValueError(
            f"No OTF file has been set for channel {wavelengths.emission_nm_int} ({wavelengths})"
        )

    otf_path = Path(
        copyfile(
            otf_path,
            output_dir / otf_path.name,
        )
    )

    kwargs = _prepare_config_kwargs(
        conf,
        emission_wavelength=wavelengths.emission_nm_int,
        otf_path=otf_path,
        **config_kwargs,
    )

    config_path = create_wavelength_config(
        output_dir / f"config{wavelengths.emission_nm_int}.txt",
        file_path,
        **kwargs,
    )
    output_path = create_output_path(
        file_path,
        output_type="recon",
        suffix=".tiff",
        output_directory=output_dir,
        wavelength=wavelengths.emission_nm_int,
        ensure_unique=True,
    )
    return ProcessingInfo(
        file_path,
        otf_path=otf_path,
        config_path=config_path,
        output_path=output_path,
        log_path=output_path.with_suffix(".log"),
        wavelengths=wavelengths,
        kwargs=kwargs,
    )


def _prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    conf: ConfigManager,
    **config_kwargs: Any,
) -> dict[int, ProcessingInfo]:
    file_path = Path(file_path)
    processing_dir = Path(processing_dir)

    image_data = get_image_data(file_path)

    # Resolution defaults to metadata values but kwargs can override
    config_kwargs["zres"] = config_kwargs.get("zres", image_data.resolution.z)

    # Assume xyres is the y pixel size (matches the assumption used in cudasirecon, allows for deskew)
    # see https://github.com/scopetools/cudasirecon/blob/main/src/cudaSirecon/mrc.h
    config_kwargs["xyres"] = config_kwargs.get("xyres", image_data.resolution.y)

    # Safer to assume they match (if not given) than use cudasirecon's default value of 0.15:
    config_kwargs["zresPSF"] = config_kwargs.get("zresPSF", config_kwargs["zres"])

    processing_info_dict: dict[int, ProcessingInfo] = dict()

    # Create TIFFs split by wavelength
    for channel in image_data.channels:
        if conf.get_channel_config(channel.wavelengths.emission_nm_int) is not None:
            try:
                split_file_path = (
                    processing_dir / f"data{channel.wavelengths.emission_nm}.tiff"
                )
                write_tiff(
                    split_file_path,
                    channel,
                    xy_pixel_size_microns=(
                        image_data.resolution.x,
                        image_data.resolution.y,
                    ),
                )

                processing_info = create_processing_info(
                    file_path=split_file_path,
                    output_dir=processing_dir,
                    wavelengths=channel.wavelengths,
                    conf=conf,
                    **config_kwargs,
                )

                if channel.wavelengths.emission_nm_int in processing_info_dict:
                    raise KeyError(
                        f"Emission wavelength {channel.wavelengths.emission_nm_int} found multiple times within {file_path}"
                    )

                processing_info_dict[channel.wavelengths.emission_nm_int] = (
                    processing_info
                )
            except Exception:
                logger.error(
                    "Failed to prepare files for channel %i (%s) of %s",
                    channel.wavelengths.emission_nm_int,
                    channel.wavelengths,
                    file_path,
                    exc_info=True,
                )
    return processing_info_dict
