from importlib import metadata

try:
    __version__ = metadata.metadata("PySIMRecon")["version"]
except metadata.PackageNotFoundError:
    __version__ = None
