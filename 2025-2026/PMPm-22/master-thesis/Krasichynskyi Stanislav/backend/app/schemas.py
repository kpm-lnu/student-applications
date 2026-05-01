from pydantic import BaseModel, Field
from typing import List


class PointDto(BaseModel):
    x: float
    y: float


class TriangleDto(BaseModel):
    a: int
    b: int
    c: int


class SimulationParams(BaseModel):
    delta_t: float = 0.1
    max_jump: float = 2.0
    max_dist_between_points: float = 10.0
    min_dist_between_points: float = 0.5
    min_delta_t: float = 1e-5
    smoothing_iterations: int = 1
    smoothing_alpha: float = 0.5
    sampling_rate: float = 0.02
    threshold: int = 2
    force_value: float = 0.1
    a_11: float = 1.0
    a_22: float = 1.0
    boundary_value: float = 1.0
    triangulation_min_angle: float = 20.0
    triangulation_max_area: float = 0.0
    triangulation_max_steiner_points: int = 0
    pressure_a: float = 1.0
    pressure_g: float = 1.0
    pressure_chi: float = 0.0
    pressure_k: float = 0.0
    pressure_dimension: int = 2


class SimulationStateDto(BaseModel):
    points: List[PointDto]
    vertices: List[PointDto]
    triangles: List[TriangleDto]
    concentration: List[float]
    pressure: List[float]
    delta_t: float


class CreateFromPointsRequest(BaseModel):
    points: List[PointDto]
    params: SimulationParams = SimulationParams()


class StepRequest(BaseModel):
    state: SimulationStateDto
    params: SimulationParams


class RunRequest(BaseModel):
    state: SimulationStateDto
    params: SimulationParams
    steps: int = Field(default=10, ge=1, le=1000)
