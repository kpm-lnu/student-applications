from model.enums import ValidationStatuses


def check_math_model_validity(
    diffusion_coefficient, adhesion_coefficient, aptosis_coefficient
) -> ValidationStatuses:

    if diffusion_coefficient < 0 or diffusion_coefficient is None:
        return ValidationStatuses.DIFFUSION_COEFFICIENT_ERROR
    elif adhesion_coefficient < 0 or adhesion_coefficient is None:
        return ValidationStatuses.ADHESION_COEFFICIENT_ERROR
    elif aptosis_coefficient < 0 or aptosis_coefficient is None:
        return ValidationStatuses.APTOSIS_COEFFICIENT_ERROR

    return ValidationStatuses.OK
