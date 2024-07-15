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
from typing import TYPE_CHECKING, NamedTuple, cast

from .utils import get_temporary_path, OTF_NAME_STUB
from .config import create_wavelength_config
from ..info import __version__

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


def read_mrc_bound_array(file_path: str | PathLike[str]) -> NDArray[Any]:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")
    logger.debug("Reading %s", file_path)
    return mrc.mrc.imread(str(file_path))  # type: ignore[reportReturnType]


@contextmanager
def write_dv(
    output_file_path: str | PathLike[str],
    array: NDArray[Any],
    header: np.recarray[int | float, np.dtype[np.int32 | np.float32]],
) -> Generator[np.recarray[int | float, np.dtype[np.int32 | np.float32]], None, None]:
    m = None
    try:
        m = mrc.Mrc2(output_file_path, mode="w")
        m.initHdrForArr(array)  # type: ignore[reportUnknownMemberType]
        mrc.copyHdrInfo(m.hdr, header)  # type: ignore[reportUnknownMemberType]

        yield m.hdr  # type: ignore[reportReturnType]

        m.seekHeader()
        m.writeHeader()
        m.writeStack(array)  # type: ignore[reportUnknownMemberType]
    finally:
        if m is not None:
            m.close()


def combine_wavelengths_dv(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    *file_paths: str | PathLike[str],
    delete: bool = False,
) -> Path:
    logger.debug(
        "Combining wavelengths to %s from:\n%s",
        output_file,
        "\n\t".join(str(fp) for fp in file_paths),
    )

    with write_dv(
        output_file,
        array=np.stack(tuple(tf.memmap(fp).squeeze() for fp in file_paths), -3),  # type: ignore[reportUnknownArgumentType]
        header=read_mrc_bound_array(input_file).Mrc.hdr,  # type: ignore[reportUnknownArgumentType]
    ):
        pass

    if delete:
        try:
            for f in file_paths:
                os.remove(f)
        except Exception:
            pass

    return Path(output_file)


def write_single_channel_dv(
    output_file_path: str | PathLike[str],
    array: NDArray[Any],
    header: np.recarray[int | float, np.dtype[np.int32 | np.float32]],
    wavelength: int,
) -> None:
    """Writes a new single-channel file from array data, copying information from hdr"""
    logger.debug("Writing channel %i to %s", wavelength, output_file_path)
    with write_dv(output_file_path, array=array, header=header) as hdr:
        hdr.wave = [wavelength, 0, 0, 0, 0]
        hdr.NumWaves = 1
    return None


def write_single_channel_tiff(
    output_file_path: str | PathLike[str],
    array: NDArray[Any],
) -> None:
    """Writes a new single-channel file from array data, copying information from hdr"""
    logger.debug("Writing single channel to %s", output_file_path)

    bigtiff = (
        array.size * array.itemsize >= np.iinfo(np.uint32).max
    )  # Check if data bigger than 4GB TIFF limit

    with tf.TiffWriter(output_file_path, mode="w", bigtiff=bigtiff) as tiff:
        tiff.write(
            array,
            photometric="MINISBLACK",
            metadata={"axes": "ZYX"},
            software=f"{__package__} {__version__}",
        )
    return None


def create_processing_files(
    file_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    wavelength: int,
    settings: SettingsManager,
    **config_kwargs: Any,
) -> ProcessingFiles | None:
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    if not file_path.is_file():
        return None
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)
    otf_path = settings.get_otf_path(wavelength)

    if otf_path is None:
        raise ValueError(f"No OTF file has been set for wavelength {wavelength}")

    otf_path = Path(
        copyfile(
            otf_path,
            output_dir
            / f"{file_path.stem}_{wavelength}_{OTF_NAME_STUB}{otf_path.suffix}",
        )
    )
    # Use the configured per-wavelength settings
    kwargs = settings.get_reconstruction_config(wavelength)
    # config_kwargs override those any config defaults set
    kwargs.update(config_kwargs)
    config_path = create_wavelength_config(
        output_dir / f"{file_path.stem}_{wavelength}.cfg",
        otf_path,
        **kwargs,
    )
    return ProcessingFiles(file_path, otf_path, config_path)


def prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    settings: SettingsManager,
    **config_kwargs: Any,
) -> dict[int, ProcessingFiles]:
    waves: tuple[int, int, int, int, int]

    file_path = Path(file_path)
    processing_dir = Path(processing_dir)
    array = read_mrc_bound_array(file_path)
    header = array.Mrc.hdr  # type: ignore[reportUnknownMemberType]
    processing_files_dict: dict[int, ProcessingFiles] = dict()
    waves = cast(tuple[int, int, int, int, int], header.wave)  # type: ignore[reportUnknownMemberType]
    # Get resolution values from DV file (they get applied to TIFFs later)
    # Resolution defaults to metadata values but kwargs can override
    config_kwargs["zres"] = config_kwargs.get("zres", header.d[2])  # type: ignore[reportUnknownMemberType]
    # Assumes square pixels:
    config_kwargs["xyres"] = config_kwargs.get("xyres", header.d[0])  # type: ignore[reportUnknownMemberType]
    if np.count_nonzero(waves) == 1:
        # if it's a single channel file, we don't need to split
        wavelength = waves[0]

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
        # otherwise break out individual wavelengths
        for c, wavelength in enumerate(waves):
            if wavelength == 0:
                continue
            processing_files = None
            if settings.get_wavelength(wavelength) is not None:
                proc_output_path = (
                    processing_dir / f"{file_path.stem}_{wavelength}.tiff"
                )
                # assumes channel is the 3rd to last dimension
                # Equivalent of np.take(array, c, -3) but no copying
                channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
                channel_slice[-3] = c
                write_single_channel_tiff(
                    proc_output_path,
                    array[*channel_slice],
                )
                processing_files = create_processing_files(
                    file_path=proc_output_path,
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
    dv_path: str | PathLike[str],
    directory: str | PathLike[str] | None = None,
    delete: bool = True,
) -> Generator[Path, None, None]:
    dv_path = Path(dv_path)
    tiff_path = None
    if directory is None:
        directory = dv_path.parent
    else:
        directory = Path(directory)
    try:
        tiff_path = get_temporary_path(directory, f".{dv_path.stem}", suffix=".tiff")

        with read_dv(dv_path) as dv:
            tf.imwrite(tiff_path, data=dv.asarray(squeeze=True))  # type: ignore[reportUnknownMemberType]
        yield tiff_path
    finally:
        if delete and tiff_path is not None:
            os.unlink(tiff_path)


def read_tiff(filepath: str | PathLike[str]) -> NDArray[Any]:
    with tf.TiffFile(filepath) as tiff:
        return tiff.asarray()
