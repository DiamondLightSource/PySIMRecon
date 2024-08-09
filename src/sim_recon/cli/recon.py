import logging

from ..main import sim_reconstruct
from .parsing import parse_recon_args
from ..progress import set_use_tqdm


def sim_recon():
    namespace, recon_kwargs = parse_recon_args()

    set_use_tqdm(namespace.use_tqdm)

    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    sim_reconstruct(
        namespace.config_path,
        namespace.output_directory,
        *namespace.sim_data_paths,
        stitch_channels=namespace.stitch_channels,
        cleanup=namespace.cleanup,
        parallel_process=namespace.parallel_process,
        **recon_kwargs,
    )


if __name__ == "__main__":
    sim_recon()
