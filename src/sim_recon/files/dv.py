from __future__ import annotations

import logging
import os
import numpy as np
import mrc
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# This file contains functions modified from https://github.com/tlambert03/mrc
# to work with writing DV file metadata


class Wavelengths(NamedTuple):
    excitation: float
    emission: float


def save_dv(
    array: NDArray[Any],
    file_path: str | PathLike[str],
    overwrite: bool = True,
    header: object = None,
    extended_header_ints: int | None = None,
    extended_header_floats: int | None = None,
    wavelengths_list: list[Wavelengths] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """reimplementation of mrc.mrc.save removing unwanted features and adding extended header"""
    if os.path.exists(file_path):
        if not overwrite:
            raise FileExistsError("Not overwriting existing file '%s'" % file_path)

    m = Mrc3(file_path, mode="w")
    m.initHdrForArr(array)
    if header is not None:
        mrc.copyHdrInfo(m.hdr, header)
    else:
        # added by Talley to detect whether array is Mrc format and copy header if so
        if hasattr(array, "Mrc"):
            if hasattr(array.Mrc, "hdr"):
                mrc.copyHdrInfo(m.hdr, array.Mrc.hdr)

    mrc.mrc.calculate_mmm(array, m)
    mrc.mrc.add_metadata(metadata, m.hdr)
    if extended_header_ints is not None or extended_header_floats is not None:
        m.insert_extended_header(extended_header_ints, extended_header_floats)
        m.update_extended_header(array, wavelengths_list)
    m.writeHeader()
    m.writeExtHeader()
    m.writeStack(array)
    m.close()


class Mrc3(mrc.Mrc2):
    """
    Not safe for general use but the axis orders are when writing from this

    Info on the DV extended header from https://microscope-cockpit.org/file-format

        The extended header has the following structure per plane

        8 32bit signed integers, often are all set to zero.

        Followed by 32 32bit floats. We only what the first 14 are:

        Float index | Meta data content
        ------------|----------------------------------------------
        0           | photosensor reading (typically in mV)
        1           | elapsed time (seconds since experiment began)
        2           | x stage coordinates
        3           | y stage coordinates
        4           | z stage coordinates
        5           | minimum intensity
        6           | maximum intensity
        7           | mean intensity
        8           | exposure time (seconds)
        9           | neutral density (fraction of 1 or percentage)
        10          | excitation wavelength
        11          | emission wavelength
        12          | intensity scaling (usually 1)
        13          | energy conversion factor (usually 1)

    """

    __DV_EXT_HDR_NAMES = (
        "photosensorReading",  # 0
        "timeStampSeconds",  # 1
        "stageXCoord",  # 2
        "stageYCoord",  # 3
        "stageZCoord",  # 4
        "minInten",  # 5
        "maxInten",  # 6
        "meanInten",  # 7
        "expTime",  # 8
        "ndFilter",  # 9
        "exWavelen",  # 10
        "emWavelen",  # 11
        "intenScaling",  # 12
        "energyConvFactor",  # 13
    )

    def insert_extended_header(self, num_ints: int, num_floats: int):
        self._extHdrNumInts = self.hdr.NumIntegers = num_ints
        self._extHdrNumFloats = self.hdr.NumFloats = num_floats
        self._extHdrBytesPerSec = (self._extHdrNumInts + self._extHdrNumFloats) * 4

        n_sectors = self._shape[0]

        self._extHdrSize = self.hdr.next = mrc.mrc.minExtHdrSize(
            n_sectors, self._extHdrBytesPerSec
        )
        self._dataOffset = self._hdrSize + self._extHdrSize
        if self._extHdrSize > 0 and (
            self._extHdrNumInts > 0 or self._extHdrNumFloats > 0
        ):
            n_sectors = int(self._extHdrSize / self._extHdrBytesPerSec)
            self._extHdrArray = np.recarray(
                shape=n_sectors,
                formats="%di4,%df4" % (self._extHdrNumInts, self._extHdrNumFloats),
                names="int,float",
            )
            self.extInts = self._extHdrArray.field("int")
            self.extFloats = self._extHdrArray.field("float")
        byteorder = "="
        _fmt = "%sf4" % (byteorder)
        dv_floats: list[tuple[str, str]] = []

        for name in Mrc3.__DV_EXT_HDR_NAMES:
            dv_floats.append((name, _fmt))
        for i in range(num_floats - len(Mrc3.__DV_EXT_HDR_NAMES)):
            dv_floats.append(("empty%d" % i, _fmt))
        dv_dtype = np.dtype(dv_floats)

        type_descr = np.dtype(
            [
                ("int", "%s%di4" % (byteorder, num_ints)),
                ("float", dv_dtype),
            ]
        )

        self._extHdrArray = np.recarray(shape=n_sectors, dtype=type_descr)
        if self._fileIsByteSwapped:
            self._extHdrArray = self._extHdrArray.newbyteorder()
        self.extInts = self._extHdrArray.field("int")
        self.extFloats = self._extHdrArray.field("float")

    def update_extended_header(
        self, array: NDArray[Any], wavelengths_list: list[Wavelengths]
    ) -> None:
        # This counts on the fact that we're saving with the z-axis order of zw
        i = 0
        for channel in array:
            for plane, wavelengths in zip(channel, wavelengths_list):
                for name in Mrc3.__DV_EXT_HDR_NAMES:
                    # This feels a bit hacky and slow
                    if name == "minInten":
                        self._extHdrArray[i]["float"][name] = np.min(plane)
                    elif name == "maxInten":
                        self._extHdrArray[i]["float"][name] = np.max(plane)
                    elif name == "meanInten":
                        self._extHdrArray[i]["float"]["meanInten"] = np.mean(plane)
                    elif name == "exWavelen":
                        self._extHdrArray[i]["float"][
                            "exWavelen"
                        ] = wavelengths.excitation
                    elif name == "emWavelen":
                        self._extHdrArray[i]["float"][
                            "emWavelen"
                        ] = wavelengths.emission
                    else:
                        # Ensure all not-set values are 0
                        # TODO: Ensure values are 0 when setting up rec array to avoid this loop
                        # TODO: Check int values
                        self._extHdrArray[i]["float"][name] = 0.0
                i += 1
