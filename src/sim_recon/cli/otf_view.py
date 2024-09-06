import logging

from ..otf_view import plot_otfs
from .parsing.otf_view import parse_args
from ..progress import set_use_tqdm


def main():
    namespace = parse_args()

    set_use_tqdm(namespace.use_tqdm)

    logging.basicConfig(level=logging.DEBUG if namespace.verbose else logging.INFO)

    plot_otfs(
        *namespace.otf_paths,
        output_directory=namespace.output_directory,
        save=not namespace.show_only,
        show=namespace.show or namespace.show_only,
    )


if __name__ == "__main__":
    main()
