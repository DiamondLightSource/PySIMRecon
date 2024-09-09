from .info import __version__


if __name__ == "__main__":
    from .main import sim_reconstruct, sim_psf_to_otf

    __all__ = ["__version__", "sim_reconstruct", "sim_psf_to_otf"]
else:
    __all__ = ["__version__"]
