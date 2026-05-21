export type SolveRequest = {
  A: number[][]
  B: number[][]
  C: number[][]
  pShifts: number[]
  qShifts: number[]
  maxIterations: number
  tolerance: number
}

export type SolveResponse = {
  solution: number[][]
  residualNorm: number
  iterations: number
  converged: boolean
  residualHistory: number[]
}

export type SolveError = {
  errors?: string[]
  detail?: string
  title?: string
}
