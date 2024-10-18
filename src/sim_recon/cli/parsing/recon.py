from __future__ import annotations
import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from ...settings.formatting import filter_out_invalid_kwargs, RECON_FORMATTERS
from .shared import (
    add_general_args,
    add_override_args_from_formatters,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence

__all__ = ("parse_args",)


def _otf_arg_conv(arg: str) -> tuple[int, Path]:
    channel_str, path_str = arg.split(":", maxsplit=1)
    try:
        channel_int = int(channel_str.strip())
    except Exception:
        channel_int = None
    try:
        path = Path(path_str)
    except Exception:
        path = None
    if channel_int is None or path is None:
        raise ValueError(
            f"Unable to parse {arg}: must be of the form '<channel integer>:<OTF path>'"
        )
    return channel_int, path


def parse_args(
    args: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        prog="sim-recon",
        description="Reconstruct SIM data",
        add_help=True,
    )
    parser.add_argument(
        "-d",
        "--data",
        dest="sim_data_paths",
        required=True,
        nargs="+",
        help="Paths to SIM data files to be reconstructed (multiple paths can be given)",
    )
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config_path",
        help="Path to the root config that specifies the paths to the OTFs and the other configs (recommended)",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        help="If specified, the output directory in which the reconstructed files will be saved (otherwise each reconstruction will be saved in the same directory as its SIM data file)",
    )
    parser.add_argument(
        "--otf",
        dest="otfs",
        type=_otf_arg_conv,
        action="append",
        help="The OTF file for a channel can be specified, which should be "
        "given as <emission wavelength in nm>:<the path to the OTF file> "
        "e.g. '--otf 525:/path/to/525_otf.tiff' (argument can be given "
        "multiple times to provide OTFs for multiple channels)",
    )
    parser.add_argument(
        "-amc",
        "--allow-missing-channels",
        dest="allow_missing_channels",
        action="store_true",
        help="If specified, attempt reconstruction of other channels in a multi-channel file if one or more are not configured",
    )
    parser.add_argument(
        "--type",
        dest="output_file_type",
        choices=["dv", "tiff"],
        default="dv",
        help="File type of output images",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If specified, files will be overwritten if they already exist (unique filenames will be used otherwise)",
    )
    parser.add_argument(
        "--no-cleanup",
        dest="cleanup",
        action="store_false",
        help="If specified, files created during the reconstruction process will not be cleaned up",
    )
    parser.add_argument(
        "--keep-split",
        dest="stitch_channels",
        action="store_false",
        help="If specified, channels will not be stitched back together after reconstruction",
    )
    parser.add_argument(
        "--parallel",
        dest="parallel_process",
        action="store_true",
        help="If specified, up to 2 processes will be run at a time",
    )

    add_general_args(parser)

    # Add arguments that override configured recon settings
    add_override_args_from_formatters(parser, RECON_FORMATTERS)

    namespace, _ = parser.parse_known_args(args)

    # Split out kwargs to be used in recon config(s)
    recon_kwargs = filter_out_invalid_kwargs(
        vars(namespace), RECON_FORMATTERS, allow_none=False
    )

    return namespace, recon_kwargs


if __name__ == "__main__":
    parse_args()
