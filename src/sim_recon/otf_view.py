from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

from .progress import get_progress_wrapper, get_logging_redirect
from .images.dv import read_dv
from .images.tiff import read_tiff
from .images.utils import interleaved_float_to_complex

if TYPE_CHECKING:
    from typing import Any
    from os import PathLike
    from numpy.typing import NDArray
    from matplotlib.figure import Figure


logger = logging.getLogger(__name__)


def plot_otfs(
    *file_paths: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    save: bool = True,
    show: bool = False,
) -> None:
    logger.info("Starting OTF plotting")
    progress_wrapper = get_progress_wrapper()
    logging_redirect = get_logging_redirect()
    with logging_redirect():
        for file_path in progress_wrapper(file_paths, desc="OTF files", unit="file"):
            try:
                otf_plot(
                    file_path,
                    output_directory=output_directory,
                    save=save,
                    show=show,
                )
            except Exception:
                logger.error("Failed to plot images from %s", file_path, exc_info=True)


def otf_plot(
    file_path: str | PathLike[str],
    output_directory: str | PathLike[str] | None = None,
    save: bool = False,
    show: bool = True,
) -> None:
    file_path = Path(file_path)
    fig = _create_otf_plots(_open_otf(file_path))
    if save:
        if output_directory is None:
            output_directory = file_path.parent
        else:
            output_directory = Path(output_directory)
        output_path = (output_directory / f"{file_path.stem}_plots").with_suffix(".jpg")
        fig.savefig(output_path)
    if show:
        fig.set_dpi(50)
        fig.tight_layout()
        plt.show(block=True)


def _open_otf(file_path: str | PathLike[str]) -> NDArray[Any]:
    logger.info("Reading %s", file_path)
    try:
        array = read_tiff(file_path)
        return interleaved_float_to_complex(array)
    except Exception:
        pass
    try:
        return read_dv(file_path).asarray(squeeze=True)
    except Exception:
        pass
    raise IOError(f"Unable to read {file_path}")


def _create_otf_plots(array: NDArray[Any]) -> Figure:
    """
    Create matplotlib plots of amplitude and phase for each OTF order.

    `array` is expected to be a stack of complex numbers, with amplitude as the real part and phase as the imaginary.

    The contents of this function are based on code by Ian Dobbie.
    """
    logger.debug("Plotting OTFs")

    plot_types = 2
    num_orders = array.shape[0]

    y_image_plot_size = 10
    ratio = array.shape[2] / array.shape[1]
    x_image_plot_size = int(round(ratio * y_image_plot_size))
    figsize = (
        x_image_plot_size * num_orders,
        y_image_plot_size * plot_types,
    )

    def gamma_adjust(image: NDArray[Any], power: float = 1) -> NDArray[Any]:
        if power <= 0:
            raise ValueError("Power must be greater than 0")
        elif power == 1:
            # No adjustment needed for 1
            return image
        negative_values = image < 0
        image[negative_values] *= -1
        gamma_adjusted = image**power
        gamma_adjusted[negative_values] *= -1
        return gamma_adjusted

    fig, axes = plt.subplots(
        nrows=plot_types,
        ncols=num_orders,
        figsize=figsize,
        dpi=300,
        sharex=True,
        sharey=True,
    )
    for i in range(num_orders):
        logger.debug("Plotting order %i amplitude", i)
        axes[0, i].set_title(f"Order {i}")
        if i == 0:
            axes[0, i].set_ylabel("Amplitude")
        axes[0, i].imshow(
            gamma_adjust(np.absolute(array[i, ::-1, :]), 0.3),
            vmin=-0.1,
            vmax=1,
            cmap="plasma",
        )
        axes[0, i].set_yticks([])
        axes[0, i].set_xticks([])

        logger.debug(f"Plotting order {i} phase")
        if i == 0:
            axes[1, i].set_ylabel("Phase")
        axes[1, i].imshow(
            gamma_adjust(np.angle(array[i, ::-1, :]), 1),
            vmin=-0.1,
            vmax=0.1,
            cmap="gray",
        )
        axes[1, i].set_yticks([])
        axes[1, i].set_xticks([])

    fig.tight_layout()
    return fig
