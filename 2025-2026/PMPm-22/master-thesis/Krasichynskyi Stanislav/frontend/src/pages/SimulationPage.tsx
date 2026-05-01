import { useMemo, useState } from "react";
import {
  createSimulationFromPoints,
  simulationRun,
  simulationStep,
} from "../api/simulationApi";
import { ControlPanel } from "../components/ControlPanel";
import { DrawingCanvas } from "../components/DrawingCanvas";
import { ResultLegend } from "../components/ResultLegend";
import { TriangulationView } from "../components/TriangulationView";
import {
  defaultSimulationParams,
  type PointDto,
  type SimulationParams,
  type SimulationStateDto,
} from "../types/simulation";

function getStats(values: number[] | undefined) {
  if (!values || values.length === 0) {
    return { min: 0, max: 1 };
  }

  return {
    min: Math.min(...values),
    max: Math.max(...values),
  };
}

export function SimulationPage() {
  const [points, setPoints] = useState<PointDto[]>([]);
  const [closed, setClosed] = useState(false);
  const [params, setParams] = useState<SimulationParams>(defaultSimulationParams);
  const [state, setState] = useState<SimulationStateDto | null>(null);
  const [stepsToRun, setStepsToRun] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showValues, setShowValues] = useState(false);

  const concentrationStats = useMemo(() => getStats(state?.concentration), [state?.concentration]);
  const pressureStats = useMemo(() => getStats(state?.pressure), [state?.pressure]);

  async function handleCreateSimulation() {
    if (points.length < 3 || !closed) return;

    try {
      setLoading(true);
      setError(null);

      const result = await createSimulationFromPoints({
        points,
        params,
      });

      setState(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Помилка створення симуляції");
    } finally {
      setLoading(false);
    }
  }

  async function handleStep() {
    if (!state) return;

    try {
      setLoading(true);
      setError(null);

      const result = await simulationStep({
        state,
        params,
      });

      setState(result);
      setPoints(result.points);
      setClosed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Помилка step");
    } finally {
      setLoading(false);
    }
  }

  async function handleRun() {
    if (!state) return;

    try {
      setLoading(true);
      setError(null);

      const result = await simulationRun({
        state,
        params,
        steps: stepsToRun,
      });

      setState(result);
      setPoints(result.points);
      setClosed(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Помилка run");
    } finally {
      setLoading(false);
    }
  }

  function handleUndoPoint() {
    setPoints((prev) => prev.slice(0, -1));
  }

  function handleClosePolygon() {
    if (points.length >= 3) {
      setClosed(true);
    }
  }

  function handleReset() {
    setPoints([]);
    setClosed(false);
    setParams(defaultSimulationParams);
    setState(null);
    setError(null);
    setStepsToRun(10);
  }

  return (
    <div className="app-shell">
      <div className="page-container">
        <header className="hero-block">
          <div>
            <h1>Симуляція рухомої області</h1>
            <p>
              Намалюй область вручну, побудуй триангуляцію, запускай обчислення і
              дивись як змінюється межа та значення у вузлах.
            </p>
          </div>
        </header>
        {error && <div className="error-box">{error}</div>}

        <div className="layout-grid">
          <ControlPanel
            params={params}
            onParamsChange={setParams}
            stepsToRun={stepsToRun}
            setStepsToRun={setStepsToRun}
            onUndoPoint={handleUndoPoint}
            onClosePolygon={handleClosePolygon}
            onCreateSimulation={handleCreateSimulation}
            onStep={handleStep}
            onRun={handleRun}
            onReset={handleReset}
            canCreate={points.length >= 3 && closed}
            canSimulate={!!state}
            closed={closed}
            loading={loading}
          />

          <div className="main-column">
            <section className="panel">
              <div className="panel-header-row">
                <div>
                  <div className="panel-title">Редактор області</div>
                  <div className="muted-text">
                    Клік — додати точку. Drag — рухати точки. Shift + drag — панорамування.
                    Колесо миші — зум.
                  </div>
                </div>
              </div>

              <DrawingCanvas points={points} onChange={setPoints} closed={closed} />

              <div className="stats-row">
                <div>Точок межі: {points.length}</div>
                <div>Полігон: {closed ? "замкнутий" : "не замкнутий"}</div>
              </div>
            </section>

            <section className="panel">
              <div className="panel-header-row">
                <div>
                  <div className="panel-title">Концентрація</div>
                  <div className="muted-text">
                    Окреме відображення сітки та поля концентрації з числовими осями.
                  </div>
                </div>

                <label className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={showValues}
                    onChange={(e) => setShowValues(e.target.checked)}
                  />
                  <span>Показувати значення у вузлах</span>
                </label>
              </div>

              <TriangulationView state={state} showValues={showValues} field="concentration" />
            </section>

            <ResultLegend
              min={concentrationStats.min}
              max={concentrationStats.max}
              title="Легенда концентрації"
            />

            <section className="panel">
              <div className="panel-header-row">
                <div>
                  <div className="panel-title">Тиск</div>
                  <div className="muted-text">
                    Поле тиску рахується окремо та виводиться поруч із концентрацією.
                  </div>
                </div>
              </div>

              <TriangulationView state={state} showValues={showValues} field="pressure" />
            </section>

            <ResultLegend
              min={pressureStats.min}
              max={pressureStats.max}
              title="Легенда тиску"
            />

            <section className="panel info-grid">
              <div>
                <div className="info-label">delta_t</div>
                <div className="info-value">{state?.delta_t ?? params.delta_t}</div>
              </div>
              <div>
                <div className="info-label">Вершини</div>
                <div className="info-value">{state?.vertices.length ?? 0}</div>
              </div>
              <div>
                <div className="info-label">Трикутники</div>
                <div className="info-value">{state?.triangles.length ?? 0}</div>
              </div>
              <div>
                <div className="info-label">Граничні точки</div>
                <div className="info-value">{state?.points.length ?? points.length}</div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
