import logging
import sys
from pathlib import Path

from ..images import dv_to_tiff
from ...exceptions import PySimReconFileNotFoundError

_logger = logging.getLogger(__name__)


def main() -> None:
    for arg in sys.argv[1:]:
        try:
            fp = Path(arg)
            if not fp.is_file():
                raise PySimReconFileNotFoundError(f"{fp} is not a file")
            dv_to_tiff(fp, fp.with_suffix(".tiff"))
        except PySimReconFileNotFoundError:
            _logger.error("File not found", exc_info=True)
        except Exception:
            _logger.error("Failed to process %s", arg, exc_info=True)


if __name__ == "__main__":
    main()
