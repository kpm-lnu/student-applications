export type PointDto = {
  x: number;
  y: number;
};

export type TriangleDto = {
  a: number;
  b: number;
  c: number;
};

export type SimulationParams = {
  delta_t: number;
  max_jump: number;
  max_dist_between_points: number;
  min_dist_between_points: number;
  min_delta_t: number;
  smoothing_iterations: number;
  smoothing_alpha: number;
  sampling_rate: number;
  threshold: number;
  force_value: number;
  a_11: number;
  a_22: number;
  boundary_value: number;
  triangulation_min_angle: number;
  triangulation_max_area: number;
  triangulation_max_steiner_points: number;
  pressure_a: number;
  pressure_g: number;
  pressure_chi: number;
  pressure_k: number;
  pressure_dimension: number;
};

export type SimulationStateDto = {
  points: PointDto[];
  vertices: PointDto[];
  triangles: TriangleDto[];
  concentration: number[];
  pressure: number[];
  delta_t: number;
};

export type CreateFromPointsRequest = {
  points: PointDto[];
  params: SimulationParams;
};

export type StepRequest = {
  state: SimulationStateDto;
  params: SimulationParams;
};

export type RunRequest = {
  state: SimulationStateDto;
  params: SimulationParams;
  steps: number;
};

export const defaultSimulationParams: SimulationParams = {
  delta_t: 0.1,
  max_jump: 2.0,
  max_dist_between_points: 20.0,
  min_dist_between_points: 5.0,
  min_delta_t: 0.00001,
  smoothing_iterations: 1,
  smoothing_alpha: 0.5,
  sampling_rate: 0.02,
  threshold: 2,
  force_value: 0.1,
  a_11: 1.0,
  a_22: 1.0,
  boundary_value: 1.0,
  triangulation_min_angle: 20,
  triangulation_max_area: 0,
  triangulation_max_steiner_points: 0,
  pressure_a: 1.0,
  pressure_g: 1.0,
  pressure_chi: 0.0,
  pressure_k: 0.0,
  pressure_dimension: 2,
};
