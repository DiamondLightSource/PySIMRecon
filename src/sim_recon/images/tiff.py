from __future__ import annotations

import logging
from pathlib import Path
import tifffile as tf
from typing import TYPE_CHECKING

from .dataclasses import ImageResolution, ImageChannel
from ..info import __version__
from ..exceptions import (
    PySimReconFileExistsError,
    UndefinedValueError,
    PySimReconIOError,
)

if TYPE_CHECKING:
    from typing import Any, Generator
    from os import PathLike
    from numpy.typing import NDArray
    from .dataclasses import Wavelengths


logger = logging.getLogger(__name__)


def check_tiff(filepath: str | PathLike[str]) -> bool:
    try:
        with tf.TiffFile(filepath):
            return True
    except tf.TiffFileError:
        pass
    return False


def read_tiff(filepath: str | PathLike[str]) -> NDArray[Any]:
    with tf.TiffFile(filepath) as tiff:
        return tiff.asarray()


def get_memmap_from_tiff(file_path: str | PathLike[str]) -> NDArray[Any]:
    try:
        return tf.memmap(file_path).squeeze()
    except Exception as e:
        logger.error("Unable to read image from %s: %s", file_path, e)
        raise


def generate_channels_from_tiffs(
    *wavelengths_path_tuple: tuple[Wavelengths, Path]
) -> Generator[ImageChannel[Wavelengths], None, None]:
    for wavelengths, fp in wavelengths_path_tuple:
        try:
            yield ImageChannel(wavelengths=wavelengths, array=get_memmap_from_tiff(fp))
        except Exception:
            raise PySimReconIOError(f"Failed to read TIFF file '{fp}'")


def write_tiff(
    output_path: str | PathLike[str],
    *channels: ImageChannel,
    resolution: ImageResolution | None = None,
    ome: bool = True,
    allow_empty_channels: bool = False,
    overwrite: bool = False,
) -> Path:
    def get_channel_dict(channel: ImageChannel) -> dict[str, Any] | None:
        channel_dict: dict[str, Any] = {}
        if channel.wavelengths is None:
            return None
        if channel.wavelengths.excitation_nm is not None:
            channel_dict["ExcitationWavelength"] = channel.wavelengths.excitation_nm
            channel_dict["ExcitationWavelengthUnits"] = "nm"
        if channel.wavelengths.emission_nm is not None:
            channel_dict["EmissionWavelength"] = channel.wavelengths.emission_nm
            channel_dict["EmissionWavelengthUnits"] = "nm"
        return channel_dict

    output_path = Path(output_path)

    logger.debug("Writing array to %s", output_path)

    if output_path.is_file():
        if overwrite:
            logger.warning("Overwriting file %s", output_path)
            output_path.unlink()
        else:
            raise PySimReconFileExistsError(f"File {output_path} already exists")

    tiff_kwargs: dict[str, Any] = {
        "software": f"PySIMRecon {__version__}",
        "photometric": "MINISBLACK",
        "metadata": {},
    }

    if resolution is not None:
        # TIFF tags:
        tiff_kwargs["resolution"] = (
            1e4 / resolution.x,
            1e4 / resolution.y,
        )
        tiff_kwargs["resolutionunit"] = (
            tf.RESUNIT.CENTIMETER
        )  # Use CENTIMETER for maximum compatibility

    if ome:
        if resolution is not None:
            # OME PhysicalSize:
            tiff_kwargs["metadata"]["PhysicalSizeX"] = resolution.x
            tiff_kwargs["metadata"]["PhysicalSizeXUnit"] = "µm"
            tiff_kwargs["metadata"]["PhysicalSizeY"] = resolution.x
            tiff_kwargs["metadata"]["PhysicalSizeYUnit"] = "µm"
            if resolution.z is not None:
                tiff_kwargs["metadata"]["PhysicalSizeZ"] = resolution.z
                tiff_kwargs["metadata"]["PhysicalSizeYUnit"] = "µm"

    with tf.TiffWriter(
        output_path,
        mode="x",  # New files only
        bigtiff=True,
        ome=ome,
        shaped=not ome,
    ) as tiff:
        for channel in channels:
            if channel.array is None:
                if allow_empty_channels:
                    logger.warning(
                        "Channel %s has no array to write",
                        channel.wavelengths,
                    )
                    continue
                raise UndefinedValueError(
                    f"{output_path} will not be created as channel {channel.wavelengths} has no array to write",
                )
            channel_kwargs = tiff_kwargs.copy()
            channel_kwargs["metadata"]["axes"] = (
                "YX" if channel.array.ndim == 2 else "ZYX"
            )
            if ome:
                channel_dict = get_channel_dict(channel)
                if channel_dict is not None:
                    channel_kwargs["metadata"]["Channel"] = channel_dict
            tiff.write(channel.array, **channel_kwargs)
    return output_path
