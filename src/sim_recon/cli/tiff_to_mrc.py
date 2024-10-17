import logging
import sys
from pathlib import Path

from ..images import tiff_to_mrc
from ..exceptions import PySimReconFileNotFoundError, PySimReconTypeError

_logger = logging.getLogger(__name__)


def main() -> None:
    for arg in sys.argv[1:]:
        try:
            fp = Path(arg)
            if not fp.is_file():
                raise PySimReconFileNotFoundError(f"{fp} is not a file")
            try:
                tiff_to_mrc(fp, fp.with_suffix(".mrc"), complex_output=True)
            except PySimReconTypeError:
                tiff_to_mrc(fp, fp.with_suffix(".mrc"), complex_output=False)
        except PySimReconFileNotFoundError:
            _logger.error("File not found", exc_info=True)
        except Exception:
            _logger.error("Failed to process %s", arg, exc_info=True)


if __name__ == "__main__":
    main()
