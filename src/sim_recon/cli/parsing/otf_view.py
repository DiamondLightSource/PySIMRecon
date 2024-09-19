from __future__ import annotations
import argparse
from typing import TYPE_CHECKING

from .shared import add_general_args

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ("parse_args",)


def parse_args(
    args: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="otf-view",
        description="Create OTF views",
        add_help=True,
    )
    parser.add_argument(
        dest="otf_paths",
        nargs="+",
        help="OTF file paths",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the plots while running",
    )
    parser.add_argument(
        "--show-only",
        dest="show_only",
        action="store_true",
        help="Show plots without saving",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory",
        help="Save to this directory if saving plots, otherwise each plot will be saved with its input file",
    )

    add_general_args(parser)

    return parser.parse_known_args(args)[0]  # throw away unknown args


if __name__ == "__main__":
    parse_args()
