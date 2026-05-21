import type { SolveError, SolveRequest, SolveResponse } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5156'

export async function solveSylvester(request: SolveRequest): Promise<SolveResponse> {
  const response = await fetch(`${API_BASE_URL}/api/sylvester/solve`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as SolveError | null
    const message = payload?.errors?.join('\n') ?? payload?.detail ?? payload?.title ?? 'Помилка API'
    throw new Error(message)
  }

  return (await response.json()) as SolveResponse
}
