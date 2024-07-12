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
    header: np.recarray,
) -> Generator[np.recarray[Any, Any], None, None]:
    m = None
    try:
        m = mrc.Mrc2(output_file_path, mode="w")
        m.initHdrForArr(array)

        # remove MRC only fields, keeping bits from initHdrForArr:
        # joint_names = set(m.hdr._array.dtype.names) & set(header._fields)
        # array_list = m.hdr._array.tolist()
        # new_mrc_header_rec = m.hdr._array[joint_names]

        # skipped_fields = ("Num", "PixelType", "next")  # from copyHdrInfo

        mrc.copyHdrInfo(m.hdr, header)

        # dv_header_rec = np.rec.fromarrays(tuple([getattr(header, f, None)] for f in header._fields), names=",".join(header._fields), shape=1)  # type: ignore[reportArgumentType])

        # if dv_header_rec.dtype.names is not None:
        #     for name in dv_header_rec.dtype.names:
        #         if name not in skipped_fields:
        #             if hasattr(m.hdr, name):
        #                 setattr(m.hdr, name, getattr(dv_header_rec, name))
        yield m.hdr  # type: ignore[reportReturnType]

        #
        m.seekHeader()
        m.writeHeader()
        # m.makeExtendedHdr(numInts=8, numFloats=32, nSecs=array.shape[0])
        # m.seekExtHeader()
        # m.writeExtHeader()
        # m.seekSec(0)
        m.writeStack(array)
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
        array=np.stack(tuple(tf.memmap(fp).squeeze() for fp in file_paths), -3),
        header=read_mrc_bound_array(input_file).Mrc.hdr,
    ):
        pass

    if delete:
        try:
            for f in file_paths:
                os.remove(f)
        except Exception:
            pass

    return Path(output_file)


def write_single_channel(
    output_file_path: str | PathLike[str],
    array: NDArray[Any],
    header: np.recarray,
    wavelength: int,
) -> None:
    """Writes a new single-channel file from array data, copying information from hdr"""
    logger.debug("Writing channel %i to %s", wavelength, output_file_path)

    # header_dict = header._asdict()
    # header_dict["wave1"] = wavelength
    # header_dict["wave2"] = 0
    # header_dict["wave3"] = 0
    # header_dict["wave4"] = 0
    # header_dict["wave5"] = 0

    # header = mrc._new.Header(**header_dict)
    with write_dv(output_file_path, array=array, header=header) as hdr:
        hdr.wave = [wavelength, 0, 0, 0, 0]
        hdr.NumWaves = 1
    return None


def create_processing_files(
    file_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    wavelength: int,
    settings: SettingsManager,
    **config_kwargs,
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
            / create_filename(
                file_path.stem,
                "OTF",
                wavelength=wavelength,
                extension=otf_path.suffix,
            ),
        )
    )
    # Use the configured per-wavelength settings
    kwargs = settings.get_reconstruction_config(wavelength)
    # config_kwargs override those any config defaults set
    kwargs.update(config_kwargs)

    # data = read_mrc_bound_array(file_path)
    # # Take from dv file, not config
    # kwargs["zres"] = data.Mrc.header.dz
    # kwargs["xyres"] = data.Mrc.header.dx
    # del data

    with read_dv(file_path) as dv:
        # Take from dv file, not config
        kwargs["zres"] = dv.hdr.dz
        kwargs["xyres"] = dv.hdr.dx
    config_path = create_wavelength_config(
        output_dir
        / f"{create_filename(file_path.stem, 'config', wavelength=wavelength, extension='.cfg')}",
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
    file_path = Path(file_path)
    processing_dir = Path(processing_dir)
    array = read_mrc_bound_array(file_path)
    header = array.Mrc.hdr
    processing_files_dict = dict()
    waves = header.wave
    # waves = (header.wave1, header.wave2, header.wave3, header.wave4, header.wave5)
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
            output_path = processing_dir / f"{wavelength}{file_path.suffix}"
            # assumes channel is the 3rd to last dimension

            if settings.get_wavelength(wavelength) is not None:

                # Equivalent of np.take(array, c, -3) but no copying
                channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
                channel_slice[-3] = c
                write_single_channel(
                    output_path, array[*channel_slice], header, wavelength
                )
                processing_files = create_processing_files(
                    file_path=output_path,
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
            tf.imwrite(tiff_path, data=dv.asarray(squeeze=True))
        yield tiff_path
    finally:
        if delete and tiff_path is not None:
            os.unlink(tiff_path)
