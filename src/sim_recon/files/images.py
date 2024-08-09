from __future__ import annotations

import logging
import os
from pathlib import Path
from shutil import copyfile
from copy import deepcopy
import numpy as np
import mrc
import tifffile as tf
from contextlib import contextmanager
from typing import TYPE_CHECKING, NamedTuple, cast

from .utils import OTF_NAME_STUB, RECON_NAME_STUB
from .config import create_wavelength_config
from ..info import __version__

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator, Collection
    from os import PathLike
    from numpy.typing import NDArray
    from ..settings import SettingsManager


logger = logging.getLogger(__name__)


class ProcessingInfo(NamedTuple):
    image_path: Path
    otf_path: Path
    config_path: Path
    output_path: Path
    kwargs: dict[str, Any]


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
    return mrc.mrc.imread(str(file_path))


def get_mrc_header_array(
    file_path: str | PathLike[str],
) -> np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]]:
    dv = read_mrc_bound_array(file_path)
    # This black magic is from the commented out bits of `makeHdrArray`.
    # Setting the memmap as a recarray, then `deepcopy`ing it allows the header
    # to be returned without requiring the large overall memmap to be kept open
    header_array = dv.Mrc.hdr._array.view()  # type: ignore[attr-defined]
    header_array.__class__ = np.recarray
    return deepcopy(
        cast(
            np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]],
            header_array,
        )
    )


def write_dv(
    input_file: str | PathLike[str],
    output_file: str | PathLike[str],
    array: NDArray[Any],
    wavelengths: Collection[int],
    zoomfact: float,
    zzoom: int,
) -> Path:
    logger.info(
        "Writing array to %s with wavelengths %s",
        output_file,
        ", ".join((str(w) for w in wavelengths)),
    )
    if len(wavelengths) != array.shape[-3]:
        raise ValueError(
            "Length of wavelengths list must be equal to the number of channels in the array"
        )
    wave = [*wavelengths, 0, 0, 0, 0, 0][:5]
    # header_array = get_mrc_header_array(input_file)
    input_data = read_mrc_bound_array(input_file)
    header = input_data.Mrc.hdr  # type: ignore
    mrc.save(
        array,
        output_file,
        hdr=header,
        metadata={
            "dx": header.d[2] / zoomfact,
            "dy": header.d[1] / zoomfact,
            "dz": header.d[0] / zzoom,
            "wave": wave,
        },
    )
    logger.info(
        "%s saved",
        output_file,
    )
    return Path(output_file)


def _prepare_config_kwargs(
    settings: SettingsManager,
    wavelength: int,
    otf_path: str | PathLike[str],
    **config_kwargs: Any,
) -> dict[str, Any]:
    # Use the configured per-wavelength settings
    kwargs = settings.get_reconstruction_config(wavelength)

    # config_kwargs override those any config defaults set
    kwargs.update(config_kwargs)

    # Set final variables:
    kwargs["wavelength"] = wavelength
    # Add otf_file that is expected by ReconParams
    # Needs to be absolute because we don't know where this might be run from
    kwargs["otf_file"] = str(Path(otf_path).absolute())

    return kwargs


def create_processing_info(
    file_path: str | PathLike[str],
    output_dir: str | PathLike[str],
    wavelength: int,
    settings: SettingsManager,
    **config_kwargs: Any,
) -> ProcessingInfo:
    file_path = Path(file_path)
    output_dir = Path(output_dir)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Cannot create processing info: file {file_path} does not exist"
        )
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)
    otf_path = settings.get_otf_path(wavelength)

    if otf_path is None:
        raise ValueError(f"No OTF file has been set for wavelength {wavelength}")

    otf_path = Path(
        copyfile(
            otf_path,
            # wavelength is already in the stem
            output_dir / f"{OTF_NAME_STUB}{wavelength}.otf",
        )
    )

    kwargs = _prepare_config_kwargs(
        settings, wavelength=wavelength, otf_path=otf_path, **config_kwargs
    )

    config_path = create_wavelength_config(
        output_dir / f"config{wavelength}.txt",
        file_path,
        **kwargs,
    )
    return ProcessingInfo(
        file_path,
        otf_path,
        config_path,
        output_dir / f"{file_path.stem}_{RECON_NAME_STUB}{file_path.suffix}",
        kwargs,
    )


def prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    settings: SettingsManager,
    **config_kwargs: Any,
) -> dict[int, ProcessingInfo]:
    waves: tuple[int, int, int, int, int]

    file_path = Path(file_path)
    processing_dir = Path(processing_dir)
    array = read_mrc_bound_array(file_path)
    header = array.Mrc.hdr  # type: ignore[attr-defined]
    processing_info_dict: dict[int, ProcessingInfo] = dict()
    waves = cast(tuple[int, int, int, int, int], header.wave)
    # Get resolution values from DV file (they get applied to TIFFs later)
    # Resolution defaults to metadata values but kwargs can override
    config_kwargs["zres"] = config_kwargs.get("zres", header.d[2])
    # Assumes square pixels:
    config_kwargs["xyres"] = config_kwargs.get("xyres", header.d[0])
    if np.count_nonzero(waves) == 1:
        # if it's a single channel file, we don't need to split
        wavelength = waves[0]

        if settings.get_wavelength(wavelength) is not None:
            processing_info = create_processing_info(
                file_path=file_path,
                output_dir=processing_dir,
                wavelength=wavelength,
                settings=settings,
                **config_kwargs,
            )
            if processing_info is None:
                logger.warning(
                    "No processing files found for '%s' at %i",
                    file_path,
                    wavelength,
                )
            else:
                processing_info_dict[wavelength] = processing_info

    else:
        # otherwise break out individual wavelengths
        for c, wavelength in enumerate(waves):
            if wavelength == 0:
                continue
            processing_info = None
            if settings.get_wavelength(wavelength) is not None:
                try:
                    split_file_path = processing_dir / f"data{wavelength}.tif"
                    # assumes channel is the 3rd to last dimension
                    # Equivalent of np.take(array, c, -3) but no copying
                    channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
                    channel_slice[-3] = c

                    write_tiff(
                        split_file_path,
                        array[*channel_slice],
                        pixel_size_microns=float(
                            config_kwargs["xyres"]  # Cast as stored as Decimal
                        ),
                        emission_wavelength_nm=wavelength,
                    )

                    processing_info = create_processing_info(
                        file_path=split_file_path,
                        output_dir=processing_dir,
                        wavelength=wavelength,
                        settings=settings,
                        **config_kwargs,
                    )

                    if wavelength in processing_info_dict:
                        raise KeyError(
                            "Wavelength %i found multiple times within %s",
                            wavelength,
                            file_path,
                        )

                    processing_info_dict[wavelength] = processing_info
                except Exception:
                    logger.error(
                        "Failed to prepare files for wavelength %i of %s",
                        wavelength,
                        file_path,
                        exc_info=True,
                    )
    return processing_info_dict


@contextmanager
def dv_to_temporary_tiff(
    dv_path: str | PathLike[str],
    tiff_path: str | PathLike[str],
    squeeze: bool = True,
    delete: bool = False,
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
) -> Generator[Path, None, None]:
    try:
        yield dv_to_tiff(
            dv_path, tiff_path, squeeze=squeeze, xy_shape=xy_shape, crop=crop
        )
    finally:
        if delete and tiff_path is not None:
            os.unlink(tiff_path)


def dv_to_tiff(
    dv_path: str | PathLike[str],
    tiff_path: str | PathLike[str],
    squeeze: bool = True,
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
) -> Path:
    with read_dv(dv_path) as dv:
        array: NDArray[Any] = dv.asarray(squeeze=squeeze)
        if xy_shape is not None:
            target_yx_shape = np.asarray(xy_shape[::-1], dtype=np.uint16)
            current_yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
            crop_amount = current_yx_shape - target_yx_shape
            min_bounds = crop_amount // 2
            max_bounds = current_yx_shape - crop_amount // 2
            array = array[
                :, min_bounds[0] : max_bounds[0], min_bounds[1] : max_bounds[1]
            ]
        elif crop > 0 and crop <= 1:
            yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
            min_bounds = np.round((yx_shape * crop) / 2).astype(np.uint16)
            max_bounds = yx_shape - min_bounds
            array = array[
                :, min_bounds[0] : max_bounds[0], min_bounds[1] : max_bounds[1]
            ]
        if np.iscomplexobj(array):
            array = array.view(np.float32)
        write_tiff(tiff_path, array)
    return Path(tiff_path)


def read_tiff(filepath: str | PathLike[str]) -> NDArray[Any]:
    with tf.TiffFile(filepath) as tiff:
        return tiff.asarray()


def get_combined_array_from_tiffs(
    *file_paths: str | PathLike[str],
) -> NDArray[Any]:
    logger.debug(
        "Combining tiffs from:\n%s",
        "\n\t".join(str(fp) for fp in file_paths),
    )
    return np.stack(tuple(tf.memmap(fp).squeeze() for fp in file_paths), -3)  # type: ignore


def write_tiff(
    output_path: str | PathLike[str],
    array: NDArray[Any],
    pixel_size_microns: float | None = None,
    excitation_wavelength_nm: float | None = None,
    emission_wavelength_nm: float | None = None,
    ome: bool = True,
) -> None:
    logger.debug("Writing array to %s", output_path)
    bigtiff = (
        array.size * array.itemsize >= np.iinfo(np.uint32).max
    )  # Check if data bigger than 4GB TIFF limit

    tiff_kwargs: dict[str, Any] = {
        "software": f"{__package__} {__version__}",
        "photometric": "MINISBLACK",
        "metadata": {"axes": "ZYX"},
    }

    if pixel_size_microns is not None:

        # TIFF tags:
        tiff_kwargs["resolution"] = (1e4 / pixel_size_microns, 1e4 / pixel_size_microns)
        tiff_kwargs["resolutionunit"] = (
            tf.RESUNIT.CENTIMETER
        )  # Use CENTIMETER for maximum compatibility

    if ome:
        if pixel_size_microns is not None:
            # OME PhysicalSize:
            tiff_kwargs["metadata"]["PhysicalSizeX"] = pixel_size_microns
            tiff_kwargs["metadata"]["PhysicalSizeY"] = pixel_size_microns
            tiff_kwargs["metadata"]["PhysicalSizeXUnit"] = "µm"
            tiff_kwargs["metadata"]["PhysicalSizeYUnit"] = "µm"

        channel_dict: dict[str, Any] = {}
        if excitation_wavelength_nm is not None:
            channel_dict["ExcitationWavelength"] = excitation_wavelength_nm
            channel_dict["ExcitationWavelengthUnits"] = "nm"
        if emission_wavelength_nm is not None:
            channel_dict["EmissionWavelength"] = emission_wavelength_nm
            channel_dict["EmissionWavelengthUnits"] = "nm"

        if channel_dict:
            tiff_kwargs["metadata"]["Channel"] = channel_dict

    with tf.TiffWriter(
        output_path, mode="w", bigtiff=bigtiff, ome=ome, shaped=not ome
    ) as tiff:
        tiff.write(array, **tiff_kwargs)
