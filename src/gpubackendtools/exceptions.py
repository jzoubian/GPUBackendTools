"""Definition of GPUBACKENDTOOLS package common exceptions"""

try:
    from exceptiongroup import ExceptionGroup
except (ImportError, ModuleNotFoundError):
    # For Python < 3.11, provide a simple ExceptionGroup fallback
    class ExceptionGroup(Exception):
        def __init__(self, message, exceptions):
            self.message = message
            self.exceptions = exceptions
            super().__init__(message)
        
        def __str__(self):
            return f"{self.message} ({len(self.exceptions)} sub-exceptions)"


class GPUBACKENDTOOLSException(Exception):
    """Base class for GPUBACKENDTOOLS package exceptions."""

    pass


class BackendUnavailableException(GPUBACKENDTOOLSException):
    """Exception raised when the backend is not available."""

    pass


class BackendNotInstalled(BackendUnavailableException):
    """Exception raised when the backend has not been installed"""
    pass


class CudaException(GPUBACKENDTOOLSException):
    """Base class for CUDA-related exceptions."""

    pass


class CuPyException(GPUBACKENDTOOLSException):
    """Base class for CuPy-related exceptions."""

    pass


class MissingDependency(GPUBACKENDTOOLSException):
    """Exception raised when a required dependency is missing."""

    pass
