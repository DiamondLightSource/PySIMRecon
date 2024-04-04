from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import mrcfile
import numpy as np
from pycudadecon import RLContext, TemporaryOTF, rl_decon


if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray


def read_mrc(file_path: str | PathLike[str]) -> NDArray[Any]:
    with mrcfile.open(file_path, permissive=True) as f:
        image_array = np.asarray(f.data)
    return image_array


# def split_channels(image_stack_array: NDArray[Any], ...) -> dict[str, NDArray[Any]]: ...


def run_recon(
    image_array: NDArray[Any], psf_array: NDArray[Any], **kwargs
) -> NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]:
    # Get any kwargs for RLContext
    context_keywords = (
        "dzdata",
        "dxdata",
        "dzpsf",
        "dxpsf",
        "deskew",
        "rotate",
        "width",
        "skewed_decon",
    )
    context_kwargs = {}
    for kw in context_keywords:
        if kw in kwargs:
            context_kwargs[kw] = kwargs.pop(kw)

    # Get any kwargs for rl_decon
    decon_keywords = (
        "background",
        "n_iters",
        "shift",
        "save_deskewed",
        "output_shape",
        "napodize",
        "nz_blend",
        "pad_val",
        "dup_rev_z",
        "skewed_decon",
    )
    decon_kwargs = {}
    for kw in decon_keywords:
        if kw in kwargs:
            decon_kwargs[kw] = kwargs.pop(kw)

    with TemporaryOTF(psf_array) as otf:
        with RLContext(image_array.shape, otf.path, **context_kwargs) as ctx:
            result = rl_decon(image_array, ctx.out_shape, **decon_kwargs)
    return result


def save_result(
    output_path: str | PathLike[str],
    result: NDArray[Any] | tuple[NDArray[Any], NDArray[Any]],
) -> None:
    output_path = Path(output_path)
    if not output_path.parent.is_dir():
        raise IOError(
            f"Cannot save to directory {output_path.parent} that does not exist"
        )
    if isinstance(result, tuple):
        deskewed_path = (
            output_path.parent / f"{output_path.stem}_deskewed{output_path.suffix}"
        )
        mrcfile.write(deskewed_path, result[1])
        result = result[0]

    mrcfile.write(output_path, result)


def run(
    output_directory: str | PathLike[str],
    psf_path: str | PathLike[str],
    *sim_data_paths: str | PathLike[str],
) -> None:
    psf_array = read_mrc(psf_path)
    output_directory = Path(output_directory)
    for sim_path in sim_data_paths:
        sim_path = Path(sim_path)
        # TODO: do channels need splitting? I'm guessing that'd take some metadata reading
        save_result(
            output_path=output_directory / sim_path.name,
            result=run_recon(read_mrc(sim_path), psf_array=psf_array),
        )


if __name__ == "__main__":
    run(*sys.argv[1:])
