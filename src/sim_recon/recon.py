from __future__ import annotations
import logging
import subprocess
import multiprocessing
import os
import traceback
from functools import partial
from os.path import abspath
from shutil import copyfile
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pycudasirecon.sim_reconstructor import SIMReconstructor, lib  # type: ignore[import-untyped]

from .files.utils import redirect_output_to, create_output_path, combine_text_files
from .files.config import create_wavelength_config
from .images import get_image_data, dv_to_tiff
from .images.dv import write_dv, image_resolution_from_mrc, read_mrc_bound_array
from .images.tiff import (
    check_tiff,
    read_tiff,
    write_tiff,
    generate_channels_from_tiffs,
)
from .images.dataclasses import (
    ImageChannel,
    ImageResolution,
    Wavelengths,
    ProcessingInfo,
    ProcessingStatus,
)
from .settings import ConfigManager
from .settings.formatting import (
    formatters_to_default_value_kwargs,
    filter_out_invalid_kwargs,
    RECON_FORMATTERS,
)
from .progress import get_progress_wrapper, get_logging_redirect
from .exceptions import (
    PySimReconException,
    PySimReconFileNotFoundError,
    ReconstructionError,
    ConfigException,
    MissingOtfException,
    UndefinedValueError,
    InvalidValueError,
)

if TYPE_CHECKING:
    from typing import Any, Literal, TypeAlias
    from os import PathLike
    from multiprocessing.pool import AsyncResult
    from numpy.typing import NDArray

    OutputFileTypes: TypeAlias = Literal["dv", "tiff"]


logger = logging.getLogger(__name__)


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
    raise ReconstructionError(f"No reconstruction file found at {expected_path}")


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
            raise ReconstructionError(
                f"Z size {z} is not divisible by the number of phases and angles ({nphases} * {ndirs})"
            )
        z = (z // z_div) * zzoom
        x = int(round(x * zoomfact))
        y = int(round(y * zoomfact))

        return _recon_get_result(reconstructor, output_shape=(z, y, x))
        # return reconstructor.get_result()

    # Rare to ever hit this as errors from the C++ often just kill the process
    except OSError as e:
        logger.error("Reconstruction failed: %s", e)
        raise ReconstructionError(
            f"Error during reconstruction with config {config_path}: {e}"
        )
    except Exception as e:
        logger.error(
            "Unexpected error during reconstruction",
            config_path,
            exc_info=True,
        )
        raise ReconstructionError(
            f"Unexpected error during reconstruction with config {config_path}: {e}"
        )


def reconstruct_from_processing_info(processing_info: ProcessingInfo) -> ProcessingInfo:
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

    rec_array = None
    with redirect_output_to(processing_info.log_path):
        sep = "-" * 80
        print(
            "\n".join(
                (
                    f"Channel {processing_info.wavelengths}",
                    "Config used:",
                    processing_info.config_path.read_text().strip(),
                    sep,
                    "The text below is the output from cudasirecon\n",
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

    if rec_array is None:
        raise ReconstructionError(
            f"No image was returned from reconstruction with {processing_info.config_path}"
        )
    elif np.isnan(rec_array).all():
        raise ReconstructionError(
            f"Empty (NaN) image was returned from reconstruction with {processing_info.config_path}"
        )

    logger.info("Reconstructed %s", processing_info.image_path)
    recon_pixel_size = float(processing_info.kwargs["xyres"]) / zoomfact
    write_tiff(
        processing_info.output_path,
        ImageChannel(processing_info.wavelengths, rec_array),
        resolution=ImageResolution(recon_pixel_size, recon_pixel_size),
        overwrite=True,
    )
    logger.debug(
        "Reconstruction of %s saved in %s",
        processing_info.image_path,
        processing_info.output_path,
    )
    processing_info.status = ProcessingStatus.COMPLETE
    return processing_info


def _reconstructions_to_output(
    sim_data_path: Path,
    file_output_directory: Path,
    processing_info_dict: dict[int, ProcessingInfo],
    stitch_channels: bool = True,
    overwrite: bool = True,
    file_type: OutputFileTypes = "dv",
) -> None:
    input_dv = read_mrc_bound_array(sim_data_path).mrc
    input_resolution = image_resolution_from_mrc(input_dv, warn_not_square=False)
    if file_type == "dv":
        suffix = ".dv"
        write_output = partial(
            write_dv,
            input_dv=input_dv,
            overwrite=overwrite,
        )
    elif file_type == "tiff":
        suffix = ".ome.tiff"
        write_output = partial(write_tiff, ome=True, overwrite=overwrite)
        del input_dv

    if stitch_channels:
        try:
            # Get zoom factors to for checking if the output can be stitched
            zoom_factors = tuple(
                (
                    processing_info.kwargs["zoomfact"],
                    processing_info.kwargs["zzoom"],
                )
                for processing_info in processing_info_dict.values()
            )

            # Check if images can be stitched
            if zoom_factors.count(zoom_factors[0]) != len(zoom_factors):
                raise InvalidValueError(
                    "Zoom factors are not consistent for all wavelengths"
                )
            # Stitch channels (if requested and possible)
            output_image_path = create_output_path(
                sim_data_path,
                output_type="recon",
                suffix=suffix,
                output_directory=file_output_directory,
                ensure_unique=not overwrite,
            )

            output_wavelengths_path_tuples = tuple(
                (pi.wavelengths, pi.output_path)
                for pi in processing_info_dict.values()
                if pi.status == ProcessingStatus.COMPLETE
            )
            if not output_wavelengths_path_tuples:
                logger.warning(
                    "No reconstructions were created from %s",
                    sim_data_path,
                )
            else:
                zoom_fact = float(zoom_factors[0][0])
                zzoom = zoom_factors[0][1]
                write_output(
                    output_image_path,
                    *generate_channels_from_tiffs(*output_wavelengths_path_tuples),
                    resolution=ImageResolution(
                        input_resolution.x / zoom_fact,
                        input_resolution.y / zoom_fact,
                        input_resolution.z / zzoom,
                    ))
            return
        except InvalidValueError as e:
            logger.warning("Unable to stitch files due to error: %s", e)

    for (
        wavelength,
        processing_info,
    ) in processing_info_dict.items():
        if processing_info.status != ProcessingStatus.COMPLETE:
            continue
        output_image_path = create_output_path(
            sim_data_path,
            output_type="recon",
            suffix=suffix,
            output_directory=file_output_directory,
            wavelength=wavelength,
            ensure_unique=not overwrite,
        )
        zoom_fact = float(processing_info.kwargs["zoomfact"])
        zzoom = processing_info.kwargs["zzoom"]
        write_output(
            output_image_path,
            *generate_channels_from_tiffs((processing_info.wavelengths, processing_info.output_path)),
            resolution=ImageResolution(
                input_resolution.x / zoom_fact,
                input_resolution.y / zoom_fact,
                input_resolution.z / zzoom,
            ),
        )


def _reconstruction_process_callback(
    processing_info: ProcessingInfo,
    wavelength: int,
    processing_info_dict: dict[int, ProcessingInfo],
) -> None:
    logger.debug("Channel %i process complete", wavelength)
    processing_info_dict[wavelength].status = processing_info.status


def _reconstruction_process_error_callback(
    exception: BaseException,
    sim_data_path: str | PathLike[str],
    wavelength: int,
    processing_info: ProcessingInfo,
) -> None:
    processing_info.status = ProcessingStatus.FAILED
    if isinstance(exception, PySimReconException):
        exception_str = str(exception)
    else:
        exception_str = "".join(traceback.format_exception(exception))
    logger.error(
        # exc_info doesn't work with the callback
        "Error occurred during reconstruction of %s channel %i: %s",
        sim_data_path,
        wavelength,
        exception_str,
    )


def _get_incomplete_channels(
    processing_info_dict: dict[int, ProcessingInfo]
) -> list[int]:
    incomplete_wavelengths: list[int] = []
    for wavelength, processing_info in processing_info_dict.items():
        if processing_info.status == ProcessingStatus.COMPLETE:
            logger.debug("%i is complete", wavelength)
        else:
            incomplete_wavelengths.append(wavelength)
            logger.warning(
                "Channel %i reconstruction ended with status '%s'",
                wavelength,
                processing_info.status,
            )
    return incomplete_wavelengths


def run_reconstructions(
    conf: ConfigManager,
    *sim_data_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None,
    overwrite: bool = False,
    cleanup: bool = False,
    stitch_channels: bool = True,
    parallel_process: bool = False,
    allow_missing_channels: bool = False,
    output_file_type: OutputFileTypes = "dv",
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
                    raise PySimReconFileNotFoundError(
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
                        allow_missing_channels=allow_missing_channels,
                        **config_kwargs,
                    )

                    try:
                        async_results: list[AsyncResult] = []
                        for wavelength, processing_info in processing_info_dict.items():
                            processing_info.status = ProcessingStatus.PENDING
                            async_results.append(
                                pool.apply_async(
                                    reconstruct_from_processing_info,
                                    args=(processing_info,),
                                    callback=partial(
                                        _reconstruction_process_callback,
                                        wavelength=wavelength,
                                        processing_info_dict=processing_info_dict,
                                    ),
                                    error_callback=partial(
                                        _reconstruction_process_error_callback,
                                        sim_data_path=sim_data_path,
                                        wavelength=wavelength,
                                        processing_info=processing_info,
                                    ),
                                )
                            )

                        # Wait for async processes to finish (otherwise there won't be files to stitch!)
                        for r in progress_wrapper(
                            async_results,
                            unit="wavelengths",
                            leave=False,
                        ):
                            r.wait()

                        incomplete_channels = _get_incomplete_channels(
                            processing_info_dict
                        )
                        if incomplete_channels and not allow_missing_channels:
                            raise ReconstructionError(
                                f"Failed to reconstruct channels: {', '.join(str(i) for i in incomplete_channels)}"
                            )

                        _reconstructions_to_output(
                            sim_data_path,
                            file_output_directory=file_output_directory,
                            processing_info_dict=processing_info_dict,
                            stitch_channels=stitch_channels,
                            overwrite=overwrite,
                            file_type=output_file_type
                        )
                    finally:
                        proc_log_files: list[Path] = []
                        for pi in processing_info_dict.values():
                            if pi.log_path.is_file():
                                proc_log_files.append(pi.log_path)
                            else:
                                logger.warning(
                                    "No reconstruction log file found for channel %s",
                                    pi.wavelengths,
                                )

                        if not proc_log_files:
                            logger.warning(
                                "No output log file created as no per-channel log files were found",
                                processing_directory,
                            )
                        else:
                            log_path = create_output_path(
                                sim_data_path,
                                output_type="recon",
                                suffix=".log",
                                output_directory=file_output_directory,
                                ensure_unique=True,
                            )
                            combine_text_files(
                                log_path,
                                *proc_log_files,
                                header="Reconstruction log",
                            )
                            logger.info(
                                "Reconstruction log file created at '%s'", log_path
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
            except ConfigException as e:
                logger.error("Unable to process %s: %s", sim_data_path, e)
            except PySimReconException as e:
                logger.error("Reconstruction failed for %s: %s", sim_data_path, e)
            except Exception:
                logger.error(
                    "Unexpected error occurred for %s", sim_data_path, exc_info=True
                )


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

    kwargs = filter_out_invalid_kwargs(kwargs, RECON_FORMATTERS, allow_none=False)

    # Set final variables:
    kwargs["wavelength"] = emission_wavelength
    # Add otf_file that is expected by ReconParams
    # Needs to be absolute because we don't know where this might be run from
    kwargs["otf_file"] = str(Path(otf_path).absolute())

    return kwargs


def _prepare_OTF_file(
    otf_path: str | PathLike[str], output_directory: str | PathLike[str]
) -> Path:
    otf_path = Path(otf_path)
    output_path = Path(output_directory) / otf_path.name
    if check_tiff(otf_path):
        return Path(
            copyfile(
                otf_path,
                output_path,
            )
        )
    return dv_to_tiff(otf_path, output_path.with_suffix(".tiff"))


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
        raise PySimReconFileNotFoundError(
            f"Cannot create processing info: file {file_path} does not exist"
        )
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)

    if wavelengths.emission_nm_int is None:
        raise UndefinedValueError(
            f"Channel is missing emission wavelength: {wavelengths} (attribute 'emission_nm_int' is None)"
        )
    otf_path = conf.get_otf_path(wavelengths.emission_nm_int)

    if otf_path is None:
        raise MissingOtfException(
            f"No OTF file has been set for channel {wavelengths.emission_nm_int} ({wavelengths})"
        )

    otf_path = _prepare_OTF_file(otf_path, output_dir)

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
    allow_missing_channels: bool = False,
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
        try:
            if conf.get_channel_config(channel.wavelengths.emission_nm_int) is None:
                raise ConfigException(
                    f"Channel {channel.wavelengths.emission_nm_int} has not been configured"
                )

            split_file_path = (
                processing_dir / f"data{channel.wavelengths.emission_nm_int}.tiff"
            )
            write_tiff(
                split_file_path,
                channel,
                resolution=image_data.resolution,
            )

            processing_info = create_processing_info(
                file_path=split_file_path,
                output_dir=processing_dir,
                wavelengths=channel.wavelengths,
                conf=conf,
                **config_kwargs,
            )

            if channel.wavelengths.emission_nm_int in processing_info_dict:
                raise ConfigException(
                    f"Emission wavelength {channel.wavelengths.emission_nm_int} found multiple times within {file_path}"
                )

            processing_info_dict[channel.wavelengths.emission_nm_int] = processing_info
        except PySimReconException as e:
            logger.error(
                "Failed to prepare files for channel %s of %s: %s",
                channel.wavelengths,
                file_path,
                e,
            )
            if not allow_missing_channels:
                raise
        except Exception:
            logger.error(
                "Unexpected error preparing files for channel %s of %s",
                channel.wavelengths,
                file_path,
                exc_info=True,
            )
            if not allow_missing_channels:
                raise
    if not processing_info_dict:
        raise ConfigException(f"No configuration found for any channel in {file_path}")
    return processing_info_dict
