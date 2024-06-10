"""This file contains functions adapted from https://github.com/scopetools/cudasirecon/blob/master/recon.py"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from shutil import copyfile
import numpy as np
import mrc
import tifffile as tf
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple

from .utils import create_filename, get_temporary_path
from .config import create_wavelength_config

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator
    from os import PathLike
    from numpy.typing import NDArray
    from ..settings import SettingsManager


logger = logging.getLogger(__name__)


class ProcessingFiles(NamedTuple):
    image_path: Path
    otf_path: Path
    config_path: Path


def read_dv(file_path: str | PathLike[str]) -> mrc.DVFile:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    return mrc.DVFile(file_path)


def combine_wavelengths_dv(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *file_paths: str | PathLike[str],
    delete: bool = False,
) -> str:
    logger.debug(
        "Combining wavelengths to %s from:\n%s",
        output_file,
        "\n\t".join(str(fp) for fp in file_paths),
    )
    with read_dv(input_file) as dv:
        hdr = dv.hdr
    m = mrc.Mrc2(output_file, mode="w")
    # Use memmap rather than asarray
    array = np.stack((tf.memmap(fp).squeeze() for fp in file_paths), -3)
    m.initHdrForArr(array)
    mrc.copyHdrInfo(m.hdr, hdr)
    m.writeHeader()
    m.writeStack(array)
    m.close()

    if delete:
        try:
            [os.remove(f) for f in file_paths]
        except Exception:
            pass

    return output_file


def write_single_channel(
    array: NDArray[Any], output_file, header, wavelength: int
) -> None:
    """Writes a new single-channel file from array data, copying information from hdr"""
    logger.debug("Writing channel %i to %s", wavelength, output_file)
    m = mrc.Mrc2(output_file, mode="w")
    m.initHdrForArr(array)
    mrc.copyHdrInfo(m.hdr, header)
    m.hdr.NumWaves = 1
    m.hdr.wave = [wavelength, 0, 0, 0, 0]
    m.writeHeader()
    m.writeStack(array)
    m.close()


def create_processing_files(
    file_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    wavelength: int,
    settings: SettingsManager,
    config_kwargs,
) -> ProcessingFiles | None:
    file_path = Path(file_path)
    if not file_path.is_file():
        return None
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)
    wavelength_settings = settings.get_wavelength(wavelength)
    otf_path = Path(
        copyfile(
            wavelength_settings.otf,
            output_path=output_dir
            / create_filename(
                file_path,
                "OTF",
                wavelength=wavelength,
                extension=wavelength_settings.otf.suffix,
            ),
        )
    )
    kwargs = wavelength_settings.config.copy()
    with read_dv(file_path) as dv:
        # Take from dv file, not config
        kwargs["zres"] = dv.hdr.dz
        kwargs["xyres"] = dv.hdr.dx
    config_path = create_wavelength_config(
        output_dir
        / f"{create_filename(file_path, 'config', wavelength=wavelength, extension='.cfg')}",
        otf_path,
        **kwargs,
    )

    return ProcessingFiles(file_path, otf_path, config_path)


def prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    settings: SettingsManager,
    config_kwargs: Any,
) -> dict[int, ProcessingFiles]:
    """Context manager that takes care of splitting the file and making sure that
    duplicated data gets cleaned up when done.
    """
    file_path = Path(file_path)
    with read_dv(file_path) as dv:
        header = dv.hdr
        processing_files_dict = dict()
        if header.NumWaves == 1:
            # if it's a single channel file, we don't need to split
            wavelength = int(header.wave[0])

            if settings.get_wavelength(wavelength) is not None:
                processing_files = create_processing_files(
                    file_path=file_path,
                    output_dir=processing_dir,
                    wavelength=wavelength,
                    settings=settings,
                    **config_kwargs,
                )
                if processing_files is None:
                    logger.warning(
                        "No processing files found for '%s' at %i",
                        file_path,
                        wavelength,
                    )
                else:
                    processing_files_dict[wavelength] = processing_files

        else:
            # otherwise break out individual wavelenghts
            for c in range(header.NumWaves):
                processing_files = None
                wavelength = header.wave[c]
                output_path = processing_dir / f"{wavelength}{file_path.suffix}"
                # assumes channel is the 3rd to last dimension
                data = np.take(dv.data.squeeze(), c, -3)
                if settings.get_wavelength(wavelength) is not None:
                    write_single_channel(data, output_path, header, wavelength)
                    processing_files = create_processing_files(
                        output_dir=processing_dir,
                        wavelength=wavelength,
                        settings=settings,
                        **config_kwargs,
                    )

                    if processing_files is None:
                        logger.warning(
                            "No processing files found for '%s' at %i",
                            file_path,
                            wavelength,
                        )
                    else:
                        if wavelength in processing_files_dict:
                            raise KeyError(
                                "Wavelength %i found multiple times within %s",
                                wavelength,
                                file_path,
                            )
                        processing_files_dict[wavelength] = processing_files
    return processing_files_dict


@contextmanager
def dv_to_temporary_tiff(
    dv_path: str | PathLike[str], delete: bool = True
) -> Generator[Path, None, None]:
    dv_path = Path(dv_path)
    try:
        tiff_path = get_temporary_path(
            dv_path.parent, f".{dv_path.stem}", suffix=".tiff"
        )

        with read_dv(dv_path) as dv:
            tf.imwrite(tiff_path, data=dv)
        yield tiff_path
    finally:
        os.unlink(tiff_path)
