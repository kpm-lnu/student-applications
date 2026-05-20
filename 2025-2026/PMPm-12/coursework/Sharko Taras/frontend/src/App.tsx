import { useMemo, useState } from 'react'
import { solveSylvester } from './api/solverApi'
import { formatMatrix, parseMatrix, parseShiftVector } from './lib/matrixParser'
import type { SolveResponse } from './types'
import './App.css'

function App() {
  const [aInput, setAInput] = useState('4, 1\n2, 3')
  const [bInput, setBInput] = useState('1, 0\n0, 2')
  const [cInput, setCInput] = useState('8, 16\n14, 24')
  const [pShiftInput, setPShiftInput] = useState('0.5, 1.0')
  const [qShiftInput, setQShiftInput] = useState('0.5, 1.0')
  const [maxIterations, setMaxIterations] = useState(200)
  const [tolerance, setTolerance] = useState(1e-8)
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState('')
  const [result, setResult] = useState<SolveResponse | null>(null)

  const residualPreview = useMemo(() => {
    if (!result) {
      return ''
    }

    const points = result.residualHistory.slice(0, 20)
    return points
      .map((value, index) => `k=${index + 1}: ${value.toExponential(3)}`)
      .join('\n')
  }, [result])

  async function onSolve() {
    setApiError('')
    setResult(null)
    setIsLoading(true)

    try {
      const request = {
        A: parseMatrix(aInput),
        B: parseMatrix(bInput),
        C: parseMatrix(cInput),
        pShifts: parseShiftVector(pShiftInput),
        qShifts: parseShiftVector(qShiftInput),
        maxIterations,
        tolerance,
      }

      const response = await solveSylvester(request)
      setResult(response)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Невідома помилка'
      setApiError(message)
    } finally {
      setIsLoading(false)
    }
  }

  function loadDemo() {
    setAInput('4, 1\n2, 3')
    setBInput('1, 0\n0, 2')
    setCInput('8, 16\n14, 24')
    setPShiftInput('0.5, 1.0')
    setQShiftInput('0.5, 1.0')
    setMaxIterations(200)
    setTolerance(1e-8)
    setApiError('')
    setResult(null)
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <p className="eyebrow">C# + F# NUMERICAL SYSTEM</p>
        <h1>ADI-розв&apos;язувач рівняння Сильвестра</h1>
        <p className="hero-text">
          Введи матриці A, B, C для задачі AX + XB = C, параметри ADI та отримай матрицю X,
          історію збіжності й норму залишку.
        </p>
      </header>

      <section className="panel grid-panel">
        <div className="field-block">
          <label htmlFor="a-matrix">Матриця A (m x m)</label>
          <textarea id="a-matrix" value={aInput} onChange={(e) => setAInput(e.target.value)} />
        </div>
        <div className="field-block">
          <label htmlFor="b-matrix">Матриця B (n x n)</label>
          <textarea id="b-matrix" value={bInput} onChange={(e) => setBInput(e.target.value)} />
        </div>
        <div className="field-block full-width">
          <label htmlFor="c-matrix">Матриця C (m x n)</label>
          <textarea id="c-matrix" value={cInput} onChange={(e) => setCInput(e.target.value)} />
        </div>
      </section>

      <section className="panel controls">
        <div className="field-inline">
          <label htmlFor="p-shifts">P-зсуви</label>
          <input id="p-shifts" value={pShiftInput} onChange={(e) => setPShiftInput(e.target.value)} />
        </div>
        <div className="field-inline">
          <label htmlFor="q-shifts">Q-зсуви</label>
          <input id="q-shifts" value={qShiftInput} onChange={(e) => setQShiftInput(e.target.value)} />
        </div>
        <div className="field-inline compact">
          <label htmlFor="max-iterations">Max ітерацій</label>
          <input
            id="max-iterations"
            type="number"
            min={1}
            value={maxIterations}
            onChange={(e) => setMaxIterations(Number(e.target.value))}
          />
        </div>
        <div className="field-inline compact">
          <label htmlFor="tolerance">Tolerance</label>
          <input
            id="tolerance"
            type="number"
            step="any"
            value={tolerance}
            onChange={(e) => setTolerance(Number(e.target.value))}
          />
        </div>

        <div className="actions">
          <button className="btn secondary" type="button" onClick={loadDemo}>
            Демо-кейс
          </button>
          <button className="btn primary" type="button" onClick={onSolve} disabled={isLoading}>
            {isLoading ? 'Розвʼязую...' : 'Розвʼязати ADI'}
          </button>
        </div>
      </section>

      {apiError ? <section className="panel error-box">{apiError}</section> : null}

      {result ? (
        <section className="panel result-grid">
          <article>
            <h2>Статус збіжності</h2>
            <ul className="stats">
              <li>Converged: {result.converged ? 'так' : 'ні'}</li>
              <li>Ітерації: {result.iterations}</li>
              <li>Норма залишку: {result.residualNorm.toExponential(6)}</li>
            </ul>
          </article>

          <article>
            <h2>Матриця X</h2>
            <pre>{formatMatrix(result.solution)}</pre>
          </article>

          <article className="full-width">
            <h2>Історія залишку (перші 20 кроків)</h2>
            <pre>{residualPreview || 'Немає даних'}</pre>
          </article>
        </section>
      ) : null}
    </main>
  )
}

export default App
