from __future__ import annotations
import logging
import subprocess
import multiprocessing
import os
from os.path import abspath
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pycudasirecon.sim_reconstructor import SIMReconstructor, lib  # type: ignore[import-untyped]

from .files.utils import redirect_output_to, RECON_NAME_STUB
from .files.images import (
    prepare_files,
    read_tiff,
    write_single_channel_tiff,
    combine_tiffs,
    write_dv,
)

from .settings import SettingsManager
from .progress import get_progress_wrapper, get_logging_redirect

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from multiprocessing.pool import AsyncResult
    from numpy.typing import NDArray

    from .files.images import ProcessingInfo


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
    lib.SR_getResult(reconstructor._ptr, _result)  # type: ignore[reportPrivateUsage]
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
        # return reconstructor.get_result()  # type: ignore[reportUnknownMemberType]
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

    with redirect_output_to(processing_info.output_path.with_suffix(".log")):
        rec_array = reconstruct(
            data, processing_info.config_path, zoomfact, zzoom, ndirs, nphases
        )

    # rec_array = subprocess_recon(
    #     processing_info.image_path,
    #     processing_info.otf_path,
    #     processing_info.config_path,
    # )
    logger.info("Reconstructed %s", processing_info.image_path)
    write_single_channel_tiff(processing_info.output_path, rec_array)
    logger.debug(
        "Reconstruction of %s saved in %s",
        processing_info.image_path,
        processing_info.output_path,
    )
    return Path(processing_info.output_path)


def run_reconstructions(
    output_directory: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    settings: SettingsManager,
    stitch_channels: bool = True,
    cleanup: bool = False,
    parallel_process: bool = False,
    **config_kwargs: Any,
) -> None:
    output_directory = Path(output_directory)

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
            sim_data_path: str | PathLike[str]
            try:
                sim_data_path = Path(sim_data_path)
                if not sim_data_path.is_file():
                    raise FileNotFoundError(
                        f"Image file {sim_data_path} does not exist"
                    )

                processing_directory: str | Path
                with TemporaryDirectory(
                    prefix="proc_",
                    suffix=f"_{sim_data_path.stem}",
                    dir=output_directory,
                    delete=cleanup,
                ) as processing_directory:
                    processing_directory = Path(processing_directory)

                    # These processing files are cleaned up by TemporaryDirectory
                    # As single-wavelength files will be used directly and we don't
                    # want to delete real input files!
                    processing_info_dict = prepare_files(
                        sim_data_path,
                        processing_directory,
                        settings=settings,
                        **config_kwargs,
                    )

                    async_results: list[AsyncResult] = []
                    wavelengths: list[int] = []
                    output_paths: list[Path] = []
                    zoom_factors: list[tuple[float, int]] = []
                    for wavelength, processing_info in processing_info_dict.items():
                        async_results.append(
                            pool.apply_async(
                                reconstruct_from_processing_info,
                                args=(processing_info,),
                                error_callback=lambda e: logger.error(
                                    "Error occurred during reconstruction",
                                    exc_info=True,
                                ),
                            )
                        )

                        # Collect values needed for stitching:
                        wavelengths.append(wavelength)
                        zoom_factors.append(
                            (
                                processing_info.kwargs["zoomfact"],
                                processing_info.kwargs["zzoom"],
                            )
                        )
                        output_paths.append(processing_info.output_path)

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
                        filename = f"{sim_data_path.stem}_{RECON_NAME_STUB}{sim_data_path.suffix}"
                        write_dv(
                            sim_data_path,
                            output_directory / filename,
                            combine_tiffs(*output_paths),
                            wavelengths=wavelengths,
                            zoomfact=float(zoom_factors[0][0]),
                            zzoom=zoom_factors[0][1],
                        )
                    else:
                        # If not stitching, then these are the result and should be in the output directory
                        logger.info(
                            "Moving reconstructed files to output directory for %s",
                            sim_data_path,
                        )
                        for (
                            wavelength,
                            processing_info,
                        ) in processing_info_dict.items():
                            write_dv(
                                sim_data_path,
                                output_directory / processing_info.output_path.name,
                                combine_tiffs(processing_info.output_path),
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
