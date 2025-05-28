class ReadinessFlags:
    def __init__(
        self,
        is_area_set: bool = False,
        is_mesh_generated: bool = False,
        is_model_set: bool = False,
    ) -> None:
        self._is_area_set = is_area_set
        self._is_mesh_generated = is_mesh_generated
        self._is_model_set = is_model_set

    @property
    def is_area_set(self) -> bool:
        return self._is_area_set

    @is_area_set.setter
    def is_area_set(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("is_area_set must be a boolean value.")
        self._is_area_set = value

    @property
    def is_mesh_generated(self) -> bool:
        return self._is_mesh_generated

    @is_mesh_generated.setter
    def is_mesh_generated(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("is_mesh_generated must be a boolean value.")
        self._is_mesh_generated = value

    @property
    def is_model_set(self) -> bool:
        return self._is_model_set

    @is_model_set.setter
    def is_model_set(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise ValueError("is_model_set must be a boolean value.")
        self._is_model_set = value

    def __str__(self) -> str:
        return (
            f"ReadinessFlags(is_area_set={self.is_area_set}, "
            f"is_mesh_generated={self.is_mesh_generated}, "
            f"is_model_set={self.is_model_set})"
        )
