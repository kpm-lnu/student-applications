"""Validation Middleware"""


from typing import cast
from sympy import sympify, symbols, Expr, Symbol
from sympy.core.sympify import SympifyError
from sympy.abc import _clash

from core.classes.common.validation_response import ValidationResponse


def validate_expression(expression: str, allowed_symbols: list[str]) -> ValidationResponse:
    """Validation function"""

    if '' in _clash:
        del _clash['']

    result: ValidationResponse = ValidationResponse({
        'valid': True,
        'message': None
    })

    try:
        symbolic_expression: Expr = sympify(
            expression,
            evaluate=False,
            locals=_clash
        )
        symbols_in_expression: list[Symbol] = cast(
            list[Symbol],
            list(symbolic_expression.free_symbols)
        )

        allowed_status: dict[str, bool] = {
            str(variable): variable in set(symbols(allowed_symbols))
            for variable in symbols_in_expression
        }

        if not all(allowed_status.values()):
            unknown_variables: list[str] = [
                variable
                for variable, allowed in allowed_status.items()
                if not allowed
            ]

            result.valid = False
            result.message = (
                'Expression contains variables that are not known to the system: ' +
                ', '.join(f'"{variable}"' for variable in unknown_variables)
            )

    except SympifyError:
        result.valid = False
        result.message = 'Expression is not a valid'

    return result
