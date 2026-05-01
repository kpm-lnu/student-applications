import type {
  CreateFromPointsRequest,
  RunRequest,
  SimulationStateDto,
  StepRequest,
} from "../types/simulation";

const API_BASE_URL = "http://localhost:8000/api";

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `HTTP ${response.status}`;

    try {
      const errorData = await response.json();
      message = errorData?.detail ?? errorData?.message ?? message;
    } catch {
    }

    throw new Error(message);
  }

  return response.json() as Promise<T>;
}

export async function createSimulationFromPoints(
  payload: CreateFromPointsRequest,
): Promise<SimulationStateDto> {
  const response = await fetch(`${API_BASE_URL}/simulation/from-points`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse<SimulationStateDto>(response);
}

export async function simulationStep(
  payload: StepRequest,
): Promise<SimulationStateDto> {
  const response = await fetch(`${API_BASE_URL}/simulation/step`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse<SimulationStateDto>(response);
}

export async function simulationRun(
  payload: RunRequest,
): Promise<SimulationStateDto> {
  const response = await fetch(`${API_BASE_URL}/simulation/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  return parseResponse<SimulationStateDto>(response);
}