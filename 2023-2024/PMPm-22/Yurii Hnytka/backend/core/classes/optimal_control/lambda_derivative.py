"""Lambda Derivative Class"""


from sympy import Symbol

from core.classes.model.equation import Equation


class LambdaDerivative:
    """Lambda Derivative"""

    name: str
    equation: Equation

    def __init__(self, name: str, compartment: Symbol, hamiltonian: Equation) -> None:
        self.name = name
        self.equation = Equation()

        self.equation.add(- hamiltonian.expression.diff(compartment))
