import logging
import sys

from ..main import sim_reconstruct
from .parsing import parse_recon_args


def sim_recon():
    namespace, recon_kwargs = parse_recon_args(*sys.argv[1:])
    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    sim_reconstruct(
        namespace.config_path,
        namespace.output_directory,
        *namespace.sim_data_paths,
        stitch_channels=namespace.stitch_channels,
        cleanup=namespace.cleanup,
        **recon_kwargs,
    )


if __name__ == "__main__":
    sim_recon()
