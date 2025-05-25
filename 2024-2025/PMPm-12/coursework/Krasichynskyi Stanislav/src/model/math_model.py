class MathModel:
    def __init__(
        self,
        diffusion_coefficient: float = 1.0,
        adhesion_measure: float = 1.0,
        apoptosis_measure: float = 1.0,
    ) -> None:
        self._diffusion_coefficient = diffusion_coefficient
        self._adhesion_measure = adhesion_measure
        self._apoptosis_measure = apoptosis_measure

    @property
    def diffusion_coefficient(self) -> float:
        return self._diffusion_coefficient

    @diffusion_coefficient.setter
    def diffusion_coefficient(self, value: float) -> None:
        if value < 0:
            raise ValueError("Diffusion coefficient must be non-negative.")
        self._diffusion_coefficient = value

    @property
    def adhesion_measure(self) -> float:
        return self._adhesion_measure

    @adhesion_measure.setter
    def adhesion_measure(self, value: float) -> None:
        if value < 0:
            raise ValueError("Adhesion measure must be non-negative.")
        self._adhesion_measure = value

    @property
    def apoptosis_measure(self) -> float:
        return self._apoptosis_measure

    @apoptosis_measure.setter
    def apoptosis_measure(self, value: float) -> None:
        if value < 0:
            raise ValueError("Apoptosis measure must be non-negative.")
        self._apoptosis_measure = value

    def __str__(self) -> str:
        return (
            f"MathModel(diffusion_coefficient={self.diffusion_coefficient}, "
            f"adhesion_measure={self.adhesion_measure}, "
            f"apoptosis_measure={self.apoptosis_measure})"
        )
