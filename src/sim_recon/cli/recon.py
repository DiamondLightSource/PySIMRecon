import logging

from ..main import sim_reconstruct
from .parsing.recon import parse_args
from ..progress import set_use_tqdm


def main() -> None:
    namespace, recon_kwargs = parse_args()

    set_use_tqdm(namespace.use_tqdm)

    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    sim_reconstruct(
        *namespace.sim_data_paths,
        config_path=namespace.config_path,
        output_directory=namespace.output_directory,
        otf_overrides={} if namespace.otfs is None else dict(namespace.otfs),
        overwrite=namespace.overwrite,
        cleanup=namespace.cleanup,
        stitch_channels=namespace.stitch_channels,
        parallel_process=namespace.parallel_process,
        allow_missing_channels=namespace.allow_missing_channels,
        output_file_type=namespace.output_file_type,
        **recon_kwargs,
    )


if __name__ == "__main__":
    main()
