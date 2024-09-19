from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from ...settings.formatting import RECON_FORMATTERS
from .shared import (
    add_general_args,
    add_override_args_from_formatters,
    namespace_extract_to_dict,
)

if TYPE_CHECKING:
    from typing import Any
    from collections.abc import Sequence

__all__ = ("parse_args",)


def parse_args(
    args: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, dict[str, Any]]:
    parser = argparse.ArgumentParser(
        prog="sim-recon",
        description="Reconstruct SIM data",
        add_help=True,
    )
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config_path",
        required=True,
        help="Path to the root config that specifies the paths to the OTFs and the other configs",
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
        "-o",
        "--output-directory",
        help="The output directory to save reconstructed files in",
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
    recon_kwargs = namespace_extract_to_dict(
        namespace, RECON_FORMATTERS, allow_none=False
    )

    return namespace, recon_kwargs


if __name__ == "__main__":
    parse_args()
