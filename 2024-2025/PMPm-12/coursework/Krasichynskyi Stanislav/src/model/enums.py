import enum


class ValidationStatuses(enum.Enum):
    OK = 1
    DIFFUSION_COEFFICIENT_ERROR = 2
    ADHESION_COEFFICIENT_ERROR = 3
    APTOSIS_COEFFICIENT_ERROR = 4
