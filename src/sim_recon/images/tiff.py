from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
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
    from typing import Any
    from collections.abc import Generator, Collection
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
    *images: Collection[ImageChannel[Wavelengths] | ImageChannel[None]],
    resolution: ImageResolution | None = None,
    ome: bool = True,
    allow_empty_channels: bool = False,
    allow_missing_channel_info: bool = False,
    overwrite: bool = False,
) -> Path:
    def get_ome_channel_dict(*channels: ImageChannel) -> dict[str, Any] | None:
        names: list[str] = []
        excitation_wavelengths: list[float | None] = []
        excitation_wavelength_units: list[str | None] = []
        emission_wavelengths: list[float | None] = []
        emission_wavelength_units: list[str | None] = []
        for i, channel in enumerate(channels, 1):
            if channel.wavelengths is None:
                if not allow_missing_channel_info:
                    raise UndefinedValueError(f"Missing wavelengths for channel {i}")
                names.append(f"Channel {i}")
                excitation_wavelengths.append(None)
                excitation_wavelength_units.append(None)
                emission_wavelengths.append(None)
                emission_wavelength_units.append(None)

            else:
                names.append(
                    f"Channel {i}"
                    if channel.wavelengths.emission_nm_int is None
                    else str(channel.wavelengths.emission_nm_int)
                )
                excitation_wavelengths.append(channel.wavelengths.excitation_nm)
                excitation_wavelength_units.append("nm")
                emission_wavelengths.append(channel.wavelengths.emission_nm)
                emission_wavelength_units.append("nm")
        return {
            "Name": names,
            "ExcitationWavelength": excitation_wavelengths,
            "ExcitationWavelengthUnits": excitation_wavelength_units,
            "EmissionWavelength": emission_wavelengths,
            "EmissionWavelengthUnits": emission_wavelength_units,
        }

    output_path = Path(output_path)

    logger.debug("Writing array to %s", output_path)

    if output_path.is_file():
        if overwrite:
            logger.warning("Overwriting file %s", output_path)
            output_path.unlink()
        else:
            raise PySimReconFileExistsError(f"File {output_path} already exists")

    tiff_metadata: dict[str, Any] = {}
    tiff_kwargs: dict[str, Any] = {
        "software": f"PySIMRecon {__version__}",
        "photometric": "MINISBLACK",
        "metadata": tiff_metadata,
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
        tiff_metadata["Name"] = output_path.name
        if resolution is not None:
            # OME PhysicalSize:
            tiff_metadata["PhysicalSizeX"] = resolution.x
            tiff_metadata["PhysicalSizeXUnit"] = "µm"
            tiff_metadata["PhysicalSizeY"] = resolution.x
            tiff_metadata["PhysicalSizeYUnit"] = "µm"
            if resolution.z is not None:
                tiff_metadata["PhysicalSizeZ"] = resolution.z
                tiff_metadata["PhysicalSizeYUnit"] = "µm"

    with tf.TiffWriter(
        output_path,
        mode="x",  # New files only
        bigtiff=True,
        ome=ome,
        shaped=not ome,
    ) as tiff:
        for channels in images:
            channels_to_write = []
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
                channels_to_write.append(channel)

            array = np.stack(
                [_.array for _ in channels], axis=0
            )  # adds another axis, even if only 1 channel

            image_kwargs = tiff_kwargs.copy()
            if array.ndim == 3:
                # In case the images are 2D
                tiff_metadata["axes"] = "CYX"
            else:
                tiff_metadata["axes"] = "CZYX"

            if ome:
                try:
                    channel_dict = get_ome_channel_dict(*channels)
                    if channel_dict is not None:
                        tiff_metadata["Channel"] = channel_dict
                except UndefinedValueError as e:
                    if not allow_missing_channel_info:
                        logger.error("Failed to write image '%s': %s", output_path, e)
                        raise
                    logger.warning(
                        "Writing without OME Channel metadata due to error: %s", e
                    )
            tiff.write(array, **image_kwargs)
    return output_path
