"""This file contains functions adapted from https://github.com/scopetools/cudasirecon/blob/master/recon.py"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager
from shutil import rmtree, copyfile
import numpy as np
import mrc
from typing import TYPE_CHECKING

from .files.main import create_filename
from .files.config import get_wavelength_config

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator
    from os import PathLike
    from numpy.typing import NDArray
    from .settings import SettingsManager


logger = logging.getLogger(__name__)


class ProcessingFiles(namedtuple):
    image_path: Path
    otf_path: Path
    config_path: Path


def combine_wavelengths_dv(
    output_file: str | PathLike[str],
    *file_paths: str | PathLike[str],
    delete: bool = False,
) -> str:
    data = [mrc.imread(fname) for fname in file_paths]
    waves = [0, 0, 0, 0, 0]
    for i, item in enumerate(data):
        waves[i] = item.Mrc.hdr.wave[0]
    hdr = data[0].Mrc.hdr
    m = mrc.Mrc2(output_file, mode="w")
    array = np.stack(data, -3)
    m.initHdrForArr(array)
    mrc.copyHdrInfo(m.hdr, hdr)
    m.hdr.NumWaves = len(file_paths)
    m.hdr.wave = waves
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
    print(f"Writing channel {wavelength}...")
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
) -> ProcessingFiles | None:
    file_path = Path(file_path)
    if not file_path.is_file():
        return None

    wavelength_settings = settings.get_wavelength(wavelength)
    otf_path = Path(
        copyfile(
            wavelength_settings.otf,
            output_path=output_dir
            / f"{create_filename(file_path, wavelength, 'OTF')}.{wavelength_settings.otf.suffix()}",
        )
    )

    config_path = get_wavelength_config(
        settings.defaults_config,
        settings.wavelengths_config,
        settings.get_wavelength(),
        output_path=output_dir
        / f"{create_filename(file_path, wavelength, 'config')}.conf",
    )

    return ProcessingFiles(file_path, otf_path, config_path)


@contextmanager
def prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    settings: SettingsManager,
    cleanup: bool = False,
) -> Generator[ProcessingFiles, None, None]:
    """Context manager that takes care of splitting the file and making sure that
    duplicated data gets cleaned up when done.
    """
    file_path = Path(file_path)
    im = mrc.imread(file_path)
    header = im.Mrc.header
    output_dir = processing_dir / file_path.stem
    output_dir.mkdir()
    try:
        if header.NumWaves == 1:
            # if it's a single channel file, we don't need to split
            wavelength = int(header.wave[0])

            if settings.get_wavelength(wavelength) is not None:
                processing_files = create_processing_files(
                    file_path=file_path,
                    output_dir=output_dir,
                    wavelength=wavelength,
                    settings=settings,
                )
                if processing_files is None:
                    logger.warning(
                        "No processing files found for '%s' at %i",
                        file_path,
                        wavelength,
                    )
                else:
                    yield processing_files

        else:
            # otherwise break out individual wavelenghts
            for c in range(header.NumWaves):
                processing_files = None
                wavelength = header.wave[c]
                output_path = output_dir / f"{wavelength}{file_path.suffix}"
                # assumes channel is the 3rd to last dimension
                data = np.take(im, c, -3)
                if settings.get_wavelength(wavelength) is not None:
                    write_single_channel(data, output_path, header, wavelength)
                    processing_files = create_processing_files(
                        output_dir=output_dir,
                        wavelength=wavelength,
                        settings=settings,
                    )

                    if processing_files is None:
                        logger.warning(
                            "No processing files found for '%s' at %i",
                            file_path,
                            wavelength,
                        )
                    else:
                        yield processing_files

    finally:
        if cleanup:
            rmtree(output_dir)
