import logging

from ..main import sim_psf_to_otf
from .parsing.otf import parse_args
from ..progress import set_use_tqdm


def main() -> None:
    namespace, otf_kwargs = parse_args()

    set_use_tqdm(namespace.use_tqdm)

    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    sim_psf_to_otf(
        *namespace.psf_paths,
        config_path=namespace.config_path,
        output_directory=namespace.output_directory,
        overwrite=namespace.overwrite,
        cleanup=namespace.cleanup,
        xy_shape=namespace.xy_shape,
        **otf_kwargs,
    )


if __name__ == "__main__":
    main()
