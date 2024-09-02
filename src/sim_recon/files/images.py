from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfile
from copy import deepcopy
import numpy as np
import mrc
import tifffile as tf
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from .utils import OTF_NAME_STUB, RECON_NAME_STUB
from .config import create_wavelength_config
from ..info import __version__

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Generator, Collection
    from os import PathLike
    from numpy.typing import NDArray
    from ..settings import ConfigManager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingInfo:
    image_path: Path
    otf_path: Path
    config_path: Path
    output_path: Path
    wavelengths: Wavelengths
    kwargs: dict[str, Any]


@dataclass(slots=True)
class ImageData:
    resolution: ImageResolution
    channels: tuple[ImageChannel]


@dataclass(slots=True)
class ImageChannel:
    array: NDArray[Any] | None = None
    wavelengths: Wavelengths | None = None


@dataclass(slots=True, frozen=True)
class ImageResolution:
    xy: float | None
    z: float | None


@dataclass(slots=True, frozen=True)
class Wavelengths:
    excitation_nm: float | None = None
    emission_nm: float | None = None
    emission_nm_int: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "emission_nm_int", int(round(self.emission_nm)))

    def __str__(self):
        return f"excitation: {self.excitation_nm}nm; emission: {self.emission_nm}nm"


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
    array = read_mrc_bound_array(file_path)
    # This black magic is from the commented out bits of `makeHdrArray`.
    # Setting the memmap as a recarray, then `deepcopy`ing it allows the header
    # to be returned without requiring the large overall memmap to be kept open
    header_array = array.Mrc.hdr._array.view()  # type: ignore[attr-defined]
    header_array.__class__ = np.recarray
    return deepcopy(
        cast(
            np.recarray[float | int | bytes, np.dtype[np.float_ | np.int_ | np.bytes_]],
            header_array,
        )
    )


def get_wavelengths_from_dv(dv: mrc.Mrc) -> Generator[Wavelengths, None, None]:
    header = dv.header
    ext_floats = dv.extFloats

    sequence = header.ImgSequence  # (0,1,2 = (ZTW or WZT or ZWT)
    num_waves = header.NumWaves
    num_times = header.NumTimes
    dz = header.Num[2] // (num_waves * num_times)

    header_shape = {
        0: (num_waves, num_times, dz),
        1: (
            num_times,
            dz,
            num_waves,
        ),
        2: (num_times, num_waves, dz),
    }[sequence]

    ext_header = ext_floats.reshape(*header_shape, -1)

    for c in range(num_waves):
        indexes = {
            0: (0, 0, dz),
            1: (
                0,
                0,
                c,
            ),
            2: (0, c, 0),
        }[sequence]
        yield Wavelengths(
            # indexed as [image frame index, float index]
            excitation_nm=ext_header[*indexes, 10],  # exWavelen index is 10
            emission_nm=ext_header[*indexes, 11],  # emWavelen index is 11
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
    conf: ConfigManager,
    emission_wavelength: int,
    otf_path: str | PathLike[str],
    **config_kwargs: Any,
) -> dict[str, Any]:
    # Use the configured per-wavelength settings
    kwargs = conf.get_reconstruction_config(emission_wavelength)

    # config_kwargs override those any config defaults set
    kwargs.update(config_kwargs)

    # Set final variables:
    kwargs["wavelength"] = emission_wavelength
    # Add otf_file that is expected by ReconParams
    # Needs to be absolute because we don't know where this might be run from
    kwargs["otf_file"] = str(Path(otf_path).absolute())

    return kwargs


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
        raise FileNotFoundError(
            f"Cannot create processing info: file {file_path} does not exist"
        )
    logger.debug("Creating processing files for %s in %s", file_path, output_dir)

    otf_path = conf.get_otf_path(wavelengths.emission_nm_int)

    if otf_path is None:
        raise ValueError(f"No OTF file has been set for channel {wavelengths}")

    otf_path = Path(
        copyfile(
            otf_path,
            # emission wavelength is already in the stem
            output_dir / f"{OTF_NAME_STUB}{wavelengths.emission_nm_int}.otf",
        )
    )

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
    return ProcessingInfo(
        file_path,
        otf_path=otf_path,
        config_path=config_path,
        output_path=output_dir
        / f"{file_path.stem}_{RECON_NAME_STUB}{file_path.suffix}",
        wavelengths=wavelengths,
        kwargs=kwargs,
    )


def prepare_files(
    file_path: str | PathLike[str],
    processing_dir: str | PathLike[str],
    conf: ConfigManager,
    **config_kwargs: Any,
) -> dict[int, ProcessingInfo]:
    file_path = Path(file_path)
    processing_dir = Path(processing_dir)

    image_data = _get_image_data(file_path)

    # Resolution defaults to metadata values but kwargs can override
    config_kwargs["zres"] = config_kwargs.get("zres", image_data.resolution.z)
    config_kwargs["xyres"] = config_kwargs.get("xyres", image_data.resolution.xy)

    processing_info_dict: dict[int, ProcessingInfo] = dict()

    if len(image_data.channels) == 1:
        # if it's a single channel file, we don't need to split
        channel = image_data.channels[0]
        if conf.get_channel_config(channel.wavelengths.emission_nm) is not None:
            processing_info = create_processing_info(
                file_path=file_path,
                output_dir=processing_dir,
                wavelengths=channel.wavelengths,
                conf=conf,
                **config_kwargs,
            )
            if processing_info is None:
                logger.warning(
                    "No processing files found for '%s' channel %s",
                    file_path,
                    channel.wavelengths,
                )
            else:
                processing_info_dict[channel.wavelengths.emission_nm_int] = (
                    processing_info
                )

    else:
        # otherwise break out individual wavelengths
        for channel in image_data.channels:
            processing_info = None
            if conf.get_channel_config(channel.wavelengths.emission_nm_int) is not None:
                try:
                    split_file_path = (
                        processing_dir / f"data{channel.wavelengths.emission_nm}.tiff"
                    )
                    write_tiff(
                        split_file_path,
                        channel,
                        pixel_size_microns=float(
                            config_kwargs["xyres"]  # Cast as stored as Decimal
                        ),
                    )

                    processing_info = create_processing_info(
                        file_path=split_file_path,
                        output_dir=processing_dir,
                        wavelengths=channel.wavelengths,
                        conf=conf,
                        **config_kwargs,
                    )

                    if channel.wavelengths.emission_nm_int in processing_info_dict:
                        raise KeyError(
                            f"Emission wavelength {channel.wavelengths.emission_nm_int} found multiple times within {file_path}"
                        )

                    processing_info_dict[channel.wavelengths.emission_nm_int] = (
                        processing_info
                    )
                except Exception:
                    logger.error(
                        "Failed to prepare files for channel %s of %s",
                        channel.wavelengths,
                        file_path,
                        exc_info=True,
                    )
    return processing_info_dict


def get_image_data(
    file_path: str | PathLike[str],
) -> ImageData:
    file_path = Path(file_path)

    array = read_mrc_bound_array(file_path)
    xyz_resolutions = array.Mrc.header.d
    sequence_order = array.Mrc.header.ImgSequence
    channels: list[ImageChannel] = []
    for c, wavelengths in enumerate(get_wavelengths_from_dv(array.Mrc)):
        channel_slice: list[slice | int] = [slice(None)] * len(array.shape)
        channel_slice[sequence_order] = c
        channels.append(
            ImageChannel(
                array[*channel_slice],
                wavelengths,
            )
        )
    return ImageData(
        channels=tuple(channels),
        # Get resolution values from DV file (they get applied to TIFFs later)
        resolution=ImageResolution(
            xy=xyz_resolutions[0], z=xyz_resolutions[2]  # Assumes square pixels
        ),
    )


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


def _apply_crop(
    array: NDArray[Any],
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
) -> NDArray[Any]:
    if xy_shape is not None:
        target_yx_shape = np.asarray(xy_shape[::-1], dtype=np.uint16)
        current_yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
        crop_amount = current_yx_shape - target_yx_shape
        min_bounds = crop_amount // 2
        max_bounds = current_yx_shape - crop_amount // 2
        array = array[:, min_bounds[0] : max_bounds[0], min_bounds[1] : max_bounds[1]]
    elif crop > 0 and crop <= 1:
        yx_shape = np.asarray(array.shape[1:], dtype=np.uint16)
        min_bounds = np.round((yx_shape * crop) / 2).astype(np.uint16)
        max_bounds = yx_shape - min_bounds
        array = array[:, min_bounds[0] : max_bounds[0], min_bounds[1] : max_bounds[1]]
    return array


def dv_to_tiff(
    dv_path: str | PathLike[str],
    tiff_path: str | PathLike[str],
    xy_shape: tuple[int, int] | None = None,
    crop: float = 0,
) -> Path:
    image_data = get_image_data(dv_path)
    for channel in image_data.channels:
        channel.array = _apply_crop(channel.array, xy_shape=xy_shape, crop=crop)

        # TIFFs cannot handle complex values
        if np.iscomplexobj(channel.array):
            channel.array = complex_to_interleaved_float(channel.array)
    write_tiff(
        tiff_path, image_data.channels, pixel_size_microns=image_data.resolution.xy
    )
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
    *channels: ImageChannel,
    pixel_size_microns: float | None = None,
    ome: bool = True,
) -> None:
    def get_channel_dict(channel: ImageChannel) -> None:
        channel_dict: dict[str, Any] = {}
        if channel.wavelengths.excitation_nm is not None:
            channel_dict["ExcitationWavelength"] = channel.wavelengths.excitation_nm
            channel_dict["ExcitationWavelengthUnits"] = "nm"
        if channel.wavelengths.emission_nm is not None:
            channel_dict["EmissionWavelength"] = channel.wavelengths.emission_nm
            channel_dict["EmissionWavelengthUnits"] = "nm"
        return channel_dict

    logger.debug("Writing array to %s", output_path)

    tiff_kwargs: dict[str, Any] = {
        "software": f"PySIMRecon {__version__}",
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

    with tf.TiffWriter(
        output_path, mode="w", bigtiff=True, ome=ome, shaped=not ome
    ) as tiff:
        for channel in channels:
            channel_kwargs = tiff_kwargs.copy()
            if ome:
                channel_kwargs["metadata"]["Channel"] = get_channel_dict(channel)
            tiff.write(channel.array, **channel_kwargs)


def complex_to_interleaved_float(
    array: NDArray[np.complexfloating],
) -> NDArray[np.float32]:
    return array.view(np.float32)


def interleaved_float_to_complex(
    array: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    return array[:, :, 0::2] + 1j * array[:, :, 1::2]
