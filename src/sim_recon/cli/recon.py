import logging

from ..main import sim_reconstruct
from .parsing.recon import parse_args
from ..progress import set_use_tqdm


def main() -> None:
    namespace, recon_kwargs = parse_args()

    set_use_tqdm(namespace.use_tqdm)

    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    sim_reconstruct(
        namespace.config_path,
        *namespace.sim_data_paths,
        output_directory=namespace.output_directory,
        overwrite=namespace.overwrite,
        cleanup=namespace.cleanup,
        stitch_channels=namespace.stitch_channels,
        parallel_process=namespace.parallel_process,
        **recon_kwargs,
    )


if __name__ == "__main__":
    main()
