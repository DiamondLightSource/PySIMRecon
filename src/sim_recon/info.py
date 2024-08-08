from importlib import metadata


__version__: str | None = None
try:
    __version__ = metadata.metadata("PySIMRecon")["version"]
except metadata.PackageNotFoundError:
    pass
