from __future__ import annotations
from copy import deepcopy
from typing import TYPE_CHECKING

import mrcfile
from mrcfile.utils import printable_string_from_bytes, mode_from_dtype

import numpy as np


if TYPE_CHECKING:
    from typing import Any, Self
    from os import PathLike
    from numpy.typing import NDArray
    from mrcfile.mrcfile import MrcFile


class SliceableMrc:
    def __init__(
        self,
        data: NDArray[Any],
        header: np.record,
        extended_header: NDArray[np.void_],
    ) -> None:
        self.data = data
        self.header = header
        self.extended_header = extended_header

    @classmethod
    def from_mrc(cls, mrc: MrcFile) -> Self:
        return cls(
            data=mrc.data,
            header=mrc.header,
            extended_header=mrc.extended_header,
        )

    def get_labels(self) -> list[str]:
        return [
            label
            for bytes_ in self.header.label[: self.header.nlabl]
            for label in printable_string_from_bytes(bytes_)
        ]

    def write(self, path: str | PathLike[str], overwrite: bool = False) -> None:
        with mrcfile.new_mmap(
            path,
            shape=self.data.shape,
            mrc_mode=mode_from_dtype(self.data.dtype),
            overwrite=overwrite,
            extended_header=self.extended_header,
            exttyp=self.header.exttyp,
        ) as mrc:
            mrc.set_data(self.data)
            for label in self.get_labels():
                mrc.add_label(label)
            mrc.update_header_from_data()
            mrc.update_header_stats()

            mrc.flush()

    def __get_header_slice(self, indexes: tuple[int, int, int]) -> NDArray[np.void_]:
        value_size_bytes = 4  # (32 / 8) first 8 are ints, rest are floats
        integer_bytes = 8 * value_size_bytes
        float_bytes = 32 * value_size_bytes
        multiplier = (integer_bytes + float_bytes) * value_size_bytes
        index_range = tuple(range(*indexes))
        new_extended_header = np.full(
            (len(index_range) * multiplier,), fill_value=0, dtype=np.void
        )
        for i, j in enumerate(index_range):
            new_extended_header[i * multiplier : (i + 1) * multiplier] = (
                self.extended_header[j * multiplier : (j + 1) * multiplier]
            )
        return new_extended_header

    def _get_header_slice(
        self, indexes: tuple[int, int, int], extended_header_size_bytes: int
    ) -> np.record:
        """Get a header "slice" based on https://www.ccpem.ac.uk/mrc_format/mrc2014.php"""
        new_z_size = len(range(*indexes))
        header = deepcopy(self.header)
        header.nz = new_z_size
        header.nsymbt = extended_header_size_bytes
        return header

    def __getitem__(self, val: int | slice) -> "SliceableMrc":
        if isinstance(val, int):
            # cast ints to slices to better handle the rest
            val = slice(val, val + 1, None)
        indexes = val.indices(len(self.data))  #  Limit the slice by the size of data
        extended_header = self.__get_header_slice(indexes)
        return SliceableMrc(
            data=self.data[val, :, :],
            header=self._get_header_slice(
                indexes, extended_header_size_bytes=len(extended_header)
            ),
            extended_header=extended_header,
        )
