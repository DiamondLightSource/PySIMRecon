from importlib import metadata

if __package__ is not None:
    __version__ = metadata.metadata(__package__)["version"]
else:
    __version__ = None
