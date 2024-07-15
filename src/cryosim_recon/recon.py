from __future__ import annotations
import logging
from os.path import abspath
from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pycudasirecon import SIMReconstructor  # type: ignore[import-untyped]

from .files.utils import RECON_NAME_STUB
from .files.images import (
    prepare_files,
    read_tiff,
    write_single_channel_tiff,
    combine_wavelengths_dv,
)

from .settings import SettingsManager
from .progress import progress_wrapper, logging_redirect

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray

    from .files.images import ProcessingFiles


logger = logging.getLogger(__name__)


def reconstruct(array: NDArray[Any], config_path: str | PathLike[str]) -> NDArray[Any]:
    reconstructor = SIMReconstructor(
        np.asarray(array), config=str(abspath(config_path))
    )
    return reconstructor.get_result()  # type: ignore[reportUnknownMemberType]


def reconstruct_from_processing_files(
    processing_files: ProcessingFiles,
    output_file: str | PathLike[str],
) -> Path:
    data = read_tiff(processing_files.image_path)
    logger.info(
        "Starting reconstruction of %s with %s",
        processing_files.image_path,
        processing_files.config_path,
    )
    # Use asarray here as I'm not sure what passing a memmap would do
    # TODO: try with memmap (i.e. `dv.data.squeeze()`)
    rec_array = reconstruct(data, processing_files.config_path)
    logger.info("Reconstructed %s", processing_files.image_path)
    write_single_channel_tiff(output_file, rec_array)
    logger.debug(
        "Reconstruction of %s saved in %s", processing_files.image_path, output_file
    )
    return Path(output_file)


def run_reconstructions(
    output_directory: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
    settings: SettingsManager,
    stitch_channels: bool = True,
    cleanup: bool = False,
    **config_kwargs: Any,
) -> None:
    output_directory = Path(output_directory)
    with logging_redirect():
        for sim_data_path in progress_wrapper(
            sim_data_paths, desc="SIM data files", unit="file"
        ):
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
                    processing_files_dict = prepare_files(
                        sim_data_path,
                        processing_directory,
                        settings=settings,
                        **config_kwargs,
                    )

                    rec_paths: list[Path] = []
                    for wavelength, processing_files in progress_wrapper(
                        processing_files_dict.items(), unit="wavelength"
                    ):
                        filename = f"{sim_data_path.stem}_{wavelength}_{RECON_NAME_STUB}{sim_data_path.suffix}"
                        output_file = processing_directory / filename
                        rec_path = reconstruct_from_processing_files(
                            processing_files,
                            output_file=output_file,
                        )
                        rec_paths.append(rec_path)
                    if stitch_channels:
                        filename = f"{sim_data_path.stem}_{RECON_NAME_STUB}{sim_data_path.suffix}"
                        combine_wavelengths_dv(
                            sim_data_path,
                            output_directory / filename,
                            *rec_paths,
                            delete=cleanup,  # Clean as you go
                        )
                    else:
                        # If not stitching, then these are the result and should be in the output directory
                        logger.info(
                            "Moving reconstructed files to output directory for %s",
                            sim_data_path,
                        )
                        for p in rec_paths:
                            p.rename(output_directory / p.name)
            except Exception:
                logger.error("Error occurred for %s", sim_data_path, exc_info=True)
