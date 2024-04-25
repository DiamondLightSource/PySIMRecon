from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import mrcfile
import tifffile as tf

from .dv import read_dv as read_dv


if TYPE_CHECKING:
    from typing import Any, Literal
    from os import PathLike
    from collections.abc import Generator
    from numpy.typing import NDArray


# @dataclass
# class ImageWithMetadata:
#     image: NDArray[Any]
#     x_size: float
#     y_size: float
#     z_size: float | None = None
#     xs: tuple[float, ...] | None = None
#     ys: tuple[float, ...] | None = None
#     zs: tuple[float, ...] | None = None
#     excitation_wavelengths: tuple[float, ...] | None = None
#     emission_wavelengths: tuple[float, ...] | None = None
#     timestamps: tuple[str, ...] | None = None
#     file_name: str | None = None

#     def get_position(
#         self, index: int
#     ) -> tuple[float | None, float | None, float | None] | None:
#         if None in (self.xs, self.ys):
#             return None
#         if len(self.xs) <= index or len(self.ys) <= index:
#             return None

#         if self.zs is None or len(self.zs) <= index:
#             z = None
#         else:
#             z = self.zs[index]

#         return (self.xs[index], self.ys[index], z)

#     # def __getitem__(self, val: int | slice):
#     #     ImageWithMetadata(
#     #         image=self.image[val, :, :],
#     #         x_size=self.x_size,
#     #         y_size=self.y_size,
#     #         z_size=self.z_size,
#     #         xs=(None if self.xs is None else self.xs[val]),
#     #         ys=(None if self.ys is None else self.ys[val]),
#     #         zs=(None if self.zs is None else self.zs[val]),
#     #         excitation_wavelengths=self.excitation_wavelengths[val],
#     #         emission_wavelengths=self.emission_wavelengths[val],
#     #         timestamps=(None if self.timestamps is None else self.timestamps[val]),
#     #         filename=None,
#     #     )

#     @property
#     def pixel_size(self) -> tuple[float, float]:
#         return (self.x_size, self.y_size)

#     @property
#     def voxel_size(self) -> tuple[float, float, float | None]:
#         return (self.x_size, self.y_size, self.z_size)

#     def split_channels(
#         extended_header: NDArray[np.void_],
#         wavelength_type: Literal["emission", "excitation"],
#     ) -> Generator[slice, None, None]:

#         channel_images = {}
#         current_wavelength = -1
#         channel_start = 0
#         wavelengths_attr = getattr(self, f"{wavelength_type}_wavelengths")
#         for plane_index in range(len(self.image)):
#             wavelength = wavelengths_attr[plane_index]
#             if wavelength != current_wavelength:
#                 if current_wavelength != -1:
#                     if current_wavelength in channel_images:
#                         raise KeyError(
#                             f"{wavelength_type.capitalize()} wavelength {current_wavelength} already exists"
#                         )
#                     yield slice(channel_start, plane_index)

#                 channel_start = plane_index
#                 current_wavelength = wavelength
#             # Add final channel
#             yield slice(channel_start, plane_index)


def create_filename(
    stem: str,
    wavelength: float,
    file_type: Literal["OTF", "split", "recon"],
    extension: Literal["TIFF", "MRC"] = "TIFF",
) -> str:
    match file_type:
        case "OTF":
            filename = "{0}_{wavelength}_otf.{extension}"
        case "split":
            filename = "{0}_{wavelength}.{extension}"
        case "recon":
            filename = "{0}_{wavelength}_recon.{extension}"
        case _:
            raise ValueError("Invalid type %s given", file_type)
    return filename.format(stem, wavelength=wavelength, suffix=extension.lower())


# def save(
#     output_path: str | PathLike[str],
#     image_with_metadata: ImageWithMetadata,
#     overwrite: bool = False,
#     **kwargs: Any,
# ):
#     output_path = Path(output_path)

#     if not output_path.parent.is_dir():
#         raise IOError(
#             f"Cannot save to directory {output_path.parent} that does not exist"
#         )
#     if not overwrite and output_path.is_file():
#         raise FileExistsError(
#             f"{output_path} already exists and overwrite is set to {overwrite}"
#         )

#     extension = output_path.suffix().lstrip(".").upper()
#     match extension:
#         case "TIFF":
#             __save_as_tiff(output_path, image_with_metadata, **kwargs)
#         case "MRC":
#             __save_as_mrc(output_path, image_with_metadata, **kwargs)
#         case "DV":
#             __save_as_mrc(output_path, image_with_metadata, **kwargs)
#         case "":
#             logging.warning("No file suffix %s given, saving as .tiff")
#             __save_as_tiff(
#                 output_path.with_suffix(".tiff"),
#                 image_with_metadata,
#                 overwrite=overwrite,
#                 **kwargs,
#             )
#         case _:
#             logging.warning("Invalid file suffix %s given, saving as .tiff")
#             __save_as_tiff(
#                 output_path.with_suffix(".tiff"),
#                 image_with_metadata,
#                 overwrite=overwrite,
#                 **kwargs,
#             )


# def __save_as_tiff(
#     output_path: str | PathLike[str],
#     image_with_metadata: ImageWithMetadata,
#     **kwargs: Any,
# ) -> None:
#     metadata_str: str | None

#     tiff_kwargs = {}
#     if metadata is None:
#         metadata_str = None
#     else:
#         meta_img = metadata.images[0]
#         meta_img.pixels.type = get_ome_pixel_type(image_with_metadata.array.dtype)
#         meta_img.name = output_path.name
#         resolution_unit = (
#             tf.RESUNIT.CENTIMETER
#         )  # Must use CENTIMETER for maximum compatibility
#         try:
#             resolution = handle_tiff_resolution(metadata, resolution_unit)
#             if tf.__version__ >= "2022.7.28":
#                 # 2022.7.28: Deprecate third resolution argument on write (use resolutionunit)
#                 tiff_kwargs["resolutionunit"] = resolution_unit
#             else:
#                 resolution.append(resolution_unit)
#             tiff_kwargs["resolution"] = tuple(resolution)
#         except Exception:
#             logging.warning(
#                 "Failed to include resolution info in tiff tags", exc_info=True
#             )
#         metadata_str = metadata.to_xml()
#     bigtiff = (
#         image_with_metadata.array.size * image_with_metadata.array.itemsize
#         >= np.iinfo(np.uint32).max
#     )  # Check if data bigger than 4GB TIFF limit

#     with tf.TiffWriter(
#         str(output_path), bigtiff=bigtiff, ome=False, imagej=False
#     ) as tif:
#         tif.write(
#             image_with_metadata.array,
#             photometric="MINISBLACK",
#             description=metadata,
#             metadata={"axes": "ZYX"},
#             **tiff_kwargs,
#         )


# def __save_as_mrc(
#     output_path: Path,
#     image_with_metadata: ImageWithMetadata,
#     **kwargs: Any,
# ) -> None:
#     mrcfile.write(
#         output_path,
#         image_with_metadata.array,
#         voxel_size=image_with_metadata.voxel_size,
#         overwrite=True,
#     )
