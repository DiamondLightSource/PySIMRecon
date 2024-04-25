from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime
import warnings
from typing import TYPE_CHECKING

import numpy as np
from mrcfile import utils  # type: ignore[import-untyped]
from mrcfile.constants import MAP_ID, VOLUME_SPACEGROUP  # type: ignore[import-untyped]

from .sliceable_mrc import SliceableMrc

if TYPE_CHECKING:
    from typing import Any, Self
    from collections.abc import Generator
    from os import PathLike
    from numpy.typing import NDArray
    from mrcfile.mrcfile import MrcFile  # type: ignore[import-untyped]


dv_header_fields = [
    # from https://github.com/tlambert03/mrc/blob/main/src/mrc/mrc.py#L1150
    ("3i4", "Num", "Number of pixels in (x, y, z) dimensions"),
    (
        "i4",
        "PixelType",
        "Data type (0=uint8, 1=int16, 2=float32, 4=complex64, 6=uint16",
    ),
    (
        "3i4",
        "mst",
        "Index of the first (col/x, row/y, section/z).  (0,0,0) by default.",
    ),
    ("3i4", "m", "Pixel Sampling intervals in the (x,y,z) dimensions. usually (1,1,1)"),
    ("3f4", "d", "Pixel spacing times sampling interval in (x, y, z) dimensions"),
    ("3f4", "angle", "Cell angle (alpha, beta, gamma) in degress.  Default (90,90,90)"),
    ("3i4", "axis", "Axis (colum, row, section).  Defaults to (1, 2, 3)"),
    ("3f4", "mmm1", "(Min, Max, Mean) of the 1st wavelength image"),
    ("i2", "type", ""),  # seems to disagree with IVE header byte format/offset
    ("i2", "nspg", "Space group number (for crystallography)"),
    ("i4", "next", "Extended header size in bytes."),
    ("i2", "dvid", "ID value (-16224)"),
    ("30i1", "blank", "unused"),
    # seems to disagree with IVE header byte format/offset
    # or at least "blank" here includes "nblank", "ntst", and "blank"
    (
        "i2",
        "NumIntegers",
        "Number of 4 byte integers stored in the extended header per section.",
    ),
    (
        "i2",
        "NumFloats",
        "Number of 4 byte floating-point numbers stored "
        "in the extended header per section.",
    ),
    (
        "i2",
        "sub",
        "Number of sub-resolution data sets stored within "
        "the image. Typically, this equals 1.",
    ),
    ("i2", "zfac", "Reduction quotient for the z axis of the sub-resolution images."),
    ("2f4", "mm2", "(Min, Max) intensity of the 2nd wavelength image."),
    ("2f4", "mm3", "(Min, Max) intensity of the 3rd wavelength image."),
    ("2f4", "mm4", "(Min, Max) intensity of the 4th wavelength image."),
    (
        "i2",
        "ImageType",
        "Image type. (Type 0 used for normal imaging, 8000 used for pupil functions)",
    ),
    ("i2", "LensNum", "Lens identification number."),
    ("i2", "n1", "Depends on the image type."),
    ("i2", "n2", "Depends on the image type."),
    ("i2", "v1", "Depends on the image type."),
    ("i2", "v2", "Depends on the image type."),
    ("2f4", "mm5", "(Min, Max) intensity of the 5th wavelength image."),
    ("i2", "NumTimes", "Number of time points."),
    ("i2", "ImgSequence", "Image axis ordering. 0=XYZTW, 1=XYWZT, 2=XYZWT."),
    ("3f4", "tilt", "(x, y, z) axis tilt angle (degrees)."),
    ("i2", "NumWaves", "Number of wavelengths."),
    ("5i2", "wave", "Wavelengths (for channel [0, 1, 2, 3, 4]), in nm."),
    (
        "3f4",
        "zxy0",
        "(z,x,y) origin, in um.",
    ),  # 20050920  ## fixed: order is z,x,y NOT x,y,z
    ("i4", "NumTitles", "Number of titles. Valid numbers are between 0 and 10."),
    ("10a80", "title", "Title 1. 80 characters long."),
]

DVHEADER_DTYPE = np.dtype([(name, form) for form, name, _ in dv_header_fields])


class SliceableDv(SliceableMrc):

    def _read_header(self):
        """Read the MRC header from the I/O stream.

        The header will be read from the current stream position, and the
        stream will be advanced by 1024 bytes.

        Raises:
            :exc:`ValueError`: If the data in the stream cannot be interpreted
                 as a valid MRC file and ``permissive`` is :data:`False`.

        Warns:
            RuntimeWarning:  If the data in the stream cannot be interpreted
                 as a valid MRC file and ``permissive`` is :data:`True`.
        """
        # Read 1024 bytes from the stream
        header_arr, bytes_read = self._read_bytearray_from_stream(
            DVHEADER_DTYPE.itemsize
        )
        if bytes_read < DVHEADER_DTYPE.itemsize:
            raise ValueError("Couldn't read enough bytes for MRC header")

        # Use a recarray to allow access to fields as attributes
        # (e.g. header.mode instead of header['mode'])
        header = (
            np.frombuffer(header_arr, dtype=DVHEADER_DTYPE)
            .reshape(())
            .view(np.recarray)
        )

        # Check the map ID to make sure this is an MRC file. The full map ID
        # should be 'MAP ', but we check only the first three bytes because
        # this is the form specified in the MRC2014 paper and is used by some
        # other software.
        if bytes(header.map)[:3] != MAP_ID[:3]:
            msg = "Map ID string not found - " "not an MRC file, or file is corrupt"
            if self._permissive:
                warnings.warn(msg, RuntimeWarning)
            else:
                raise ValueError(msg)

        # Read the machine stamp to get the file's byte order
        try:
            byte_order = utils.byte_order_from_machine_stamp(header.machst)
        except ValueError as err:
            if self._permissive:
                byte_order = "<"  # try little-endian as a sensible default
                warnings.warn(str(err), RuntimeWarning)
            else:
                raise

        # Create a new dtype with the correct byte order and update the header
        header.dtype = header.dtype.newbyteorder(byte_order)

        # Check mode is valid; if not, try the opposite byte order
        # (Some MRC files have been seen 'in the wild' that are correct except
        # that the machine stamp indicates the wrong byte order.)
        if self._permissive:
            try:
                utils.dtype_from_mode(header.mode)
            except ValueError:
                try:
                    utils.dtype_from_mode(header.mode.newbyteorder())
                    # If we get here the new byte order is probably correct
                    # Use it and issue a warning
                    header.dtype = header.dtype.newbyteorder()
                    pretty_machst = utils.pretty_machine_stamp(header.machst)
                    msg = "Machine stamp '{0}' does not match the apparent byte order '{1}'"
                    warnings.warn(
                        msg.format(pretty_machst, header.mode.dtype.byteorder),
                        RuntimeWarning,
                    )
                except ValueError:
                    # Neither byte order gives a valid mode. Ignore for now,
                    # and a warning will be issued by _read_data()
                    pass

        header.flags.writeable = not self._read_only
        self._header = header

    def _create_default_header(self):
        """Create a default MRC file header.

        The header is initialised with standard file type and version
        information, default values for some essential fields, and zeros
        elsewhere. The first text label is also set to indicate the file was
        created by this module.
        """
        self._header = np.zeros(shape=(), dtype=DVHEADER_DTYPE).view(np.recarray)
        header = self._header
        header.map = MAP_ID
        header.nversion = 20141  # current MRC 2014 format version
        header.machst = utils.machine_stamp_from_byte_order(header.mode.dtype.byteorder)

        # Default space group is P1
        header.ispg = VOLUME_SPACEGROUP

        # Standard cell angles all 90.0 degrees
        default_cell_angle = 90.0
        header.cellb.alpha = default_cell_angle
        header.cellb.beta = default_cell_angle
        header.cellb.gamma = default_cell_angle
        # (this can also be achieved by assigning a 3-tuple to header.cellb
        # directly but using the sub-fields individually is easier to read and
        # understand)

        # Standard axes: columns = X, rows = Y, sections = Z
        header.mapc = 1
        header.mapr = 2
        header.maps = 3

        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header.label[0] = "{0:40s}{1:>39s} ".format("Created by mrcfile.py", time)
        header.nlabl = 1

        self.reset_header_stats()
