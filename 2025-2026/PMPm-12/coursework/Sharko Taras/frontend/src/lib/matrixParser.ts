export function parseMatrix(input: string): number[][] {
  const rows = input
    .split(/\r?\n|;/)
    .map((row) => row.trim())
    .filter(Boolean)

  if (rows.length === 0) {
    throw new Error('Матриця не може бути порожньою.')
  }

  const matrix = rows.map((row) =>
    row
      .split(/[\s,]+/)
      .map((value) => value.trim())
      .filter(Boolean)
      .map((value) => {
        const parsed = Number(value)
        if (!Number.isFinite(parsed)) {
          throw new Error(`Некоректне число: ${value}`)
        }
        return parsed
      }),
  )

  const cols = matrix[0].length
  if (cols === 0 || matrix.some((row) => row.length !== cols)) {
    throw new Error('Усі рядки матриці мають бути однакової довжини.')
  }

  return matrix
}

export function parseShiftVector(input: string): number[] {
  const values = input
    .split(/[\s,;]+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .map((value) => {
      const parsed = Number(value)
      if (!Number.isFinite(parsed)) {
        throw new Error(`Некоректний зсув: ${value}`)
      }
      return parsed
    })

  if (values.length === 0) {
    throw new Error('Потрібно задати хоча б один зсув.')
  }

  return values
}

export function formatMatrix(matrix: number[][], digits = 6): string {
  return matrix
    .map((row) => row.map((value) => value.toFixed(digits)).join('  '))
    .join('\n')
}
