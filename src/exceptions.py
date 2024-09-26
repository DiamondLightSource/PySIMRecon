class PySimReconException(Exception):
    pass


class PySimReconOSError(OSError, PySimReconException):
    pass


class PySimReconTypeError(TypeError, PySimReconException):
    pass


class PySimReconValueError(ValueError, PySimReconException):
    pass


class PySimReconFileExistsError(FileExistsError, PySimReconException):
    pass


class PySimReconFileNotFoundError(FileNotFoundError, PySimReconException):
    pass


class PySimReconIOError(IOError, PySimReconException):
    pass


class ProcessingException(PySimReconException):
    pass


class OtfError(ProcessingException):
    pass


class ReconstructionError(ProcessingException):
    pass


class ConfigException(PySimReconException):
    pass


class UndefinedValueError(PySimReconValueError):
    pass


class InvalidValueError(PySimReconValueError):
    pass


class MissingOtfException(ConfigException):
    pass
