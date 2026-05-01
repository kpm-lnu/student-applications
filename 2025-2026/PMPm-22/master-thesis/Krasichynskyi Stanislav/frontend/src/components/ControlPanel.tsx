import type { SimulationParams } from "../types/simulation";

type ControlPanelProps = {
  params: SimulationParams;
  onParamsChange: (params: SimulationParams) => void;
  stepsToRun: number;
  setStepsToRun: (value: number) => void;
  onUndoPoint: () => void;
  onClosePolygon: () => void;
  onCreateSimulation: () => void;
  onStep: () => void;
  onRun: () => void;
  onReset: () => void;
  canCreate: boolean;
  canSimulate: boolean;
  closed: boolean;
  loading: boolean;
};

export function ControlPanel({
  params,
  onParamsChange,
  stepsToRun,
  setStepsToRun,
  onUndoPoint,
  onClosePolygon,
  onCreateSimulation,
  onStep,
  onRun,
  onReset,
  canCreate,
  canSimulate,
  closed,
  loading,
}: ControlPanelProps) {
  function updateParam<K extends keyof SimulationParams>(
    key: K,
    value: number,
  ) {
    onParamsChange({
      ...params,
      [key]: value,
    });
  }

  return (
    <div className="panel sticky-panel">
      <div className="panel-title">Параметри</div>

      <div className="subsection-title">Рух межі</div>
      <div className="form-grid">
        <label>
          <span>delta_t</span>
          <input type="number" step="0.01" value={params.delta_t} onChange={(e) => updateParam("delta_t", Number(e.target.value))} />
        </label>
        <label>
          <span>max_jump</span>
          <input type="number" step="0.1" value={params.max_jump} onChange={(e) => updateParam("max_jump", Number(e.target.value))} />
        </label>
        <label>
          <span>max_dist_between_points</span>
          <input type="number" step="0.1" value={params.max_dist_between_points} onChange={(e) => updateParam("max_dist_between_points", Number(e.target.value))} />
        </label>
        <label>
          <span>min_dist_between_points</span>
          <input type="number" step="0.1" value={params.min_dist_between_points} onChange={(e) => updateParam("min_dist_between_points", Number(e.target.value))} />
        </label>
        <label>
          <span>min_delta_t</span>
          <input type="number" step="0.00001" value={params.min_delta_t} onChange={(e) => updateParam("min_delta_t", Number(e.target.value))} />
        </label>
        <label>
          <span>smoothing_iterations</span>
          <input type="number" step="1" value={params.smoothing_iterations} onChange={(e) => updateParam("smoothing_iterations", Number(e.target.value))} />
        </label>
        <label>
          <span>smoothing_alpha</span>
          <input type="number" step="0.1" value={params.smoothing_alpha} onChange={(e) => updateParam("smoothing_alpha", Number(e.target.value))} />
        </label>
      </div>

      <div className="subsection-title">Концентрація</div>
      <div className="form-grid">
        <label>
          <span>force_value</span>
          <input type="number" step="0.1" value={params.force_value} onChange={(e) => updateParam("force_value", Number(e.target.value))} />
        </label>
        <label>
          <span>a_11</span>
          <input type="number" step="0.1" value={params.a_11} onChange={(e) => updateParam("a_11", Number(e.target.value))} />
        </label>
        <label>
          <span>a_22</span>
          <input type="number" step="0.1" value={params.a_22} onChange={(e) => updateParam("a_22", Number(e.target.value))} />
        </label>
        <label>
          <span>boundary_value</span>
          <input type="number" step="0.1" value={params.boundary_value} onChange={(e) => updateParam("boundary_value", Number(e.target.value))} />
        </label>
      </div>

      <div className="subsection-title">Тиск</div>
      <div className="form-grid">
        <label>
          <span>pressure_a</span>
          <input type="number" step="0.1" value={params.pressure_a} onChange={(e) => updateParam("pressure_a", Number(e.target.value))} />
        </label>
        <label>
          <span>pressure_g</span>
          <input type="number" step="0.1" value={params.pressure_g} onChange={(e) => updateParam("pressure_g", Number(e.target.value))} />
        </label>
        <label>
          <span>pressure_chi</span>
          <input type="number" step="0.1" value={params.pressure_chi} onChange={(e) => updateParam("pressure_chi", Number(e.target.value))} />
        </label>
        <label>
          <span>pressure_k</span>
          <input type="number" step="0.1" value={params.pressure_k} onChange={(e) => updateParam("pressure_k", Number(e.target.value))} />
        </label>
        <label>
          <span>pressure_dimension</span>
          <input type="number" step="1" min={1} value={params.pressure_dimension} onChange={(e) => updateParam("pressure_dimension", Number(e.target.value))} />
        </label>
      </div>

      <div className="subsection-title">Триангуляція</div>
      <div className="form-grid">
        <label>
          <span>min_angle</span>
          <input type="number" step="1" value={params.triangulation_min_angle} onChange={(e) => updateParam("triangulation_min_angle", Number(e.target.value))} />
        </label>
        <label>
          <span>max_area (0 = off)</span>
          <input type="number" step="0.1" value={params.triangulation_max_area} onChange={(e) => updateParam("triangulation_max_area", Number(e.target.value))} />
        </label>
        <label>
          <span>max_steiner_points</span>
          <input type="number" step="1" value={params.triangulation_max_steiner_points} onChange={(e) => updateParam("triangulation_max_steiner_points", Number(e.target.value))} />
        </label>
      </div>

      <div className="subsection-title">Інше</div>
      <div className="form-grid">
        <label>
          <span>sampling_rate</span>
          <input type="number" step="0.01" value={params.sampling_rate} onChange={(e) => updateParam("sampling_rate", Number(e.target.value))} />
        </label>
        <label>
          <span>threshold</span>
          <input type="number" step="0.1" value={params.threshold} onChange={(e) => updateParam("threshold", Number(e.target.value))} />
        </label>
        <label>
          <span>steps для run</span>
          <input type="number" min={1} value={stepsToRun} onChange={(e) => setStepsToRun(Math.max(1, Number(e.target.value)))} />
        </label>
      </div>

      <div className="button-group">
        <button type="button" onClick={onUndoPoint} disabled={loading || closed}>Видалити останню точку</button>
        <button type="button" onClick={onClosePolygon} disabled={loading || closed}>Замкнути полігон</button>
        <button type="button" onClick={onCreateSimulation} disabled={loading || !canCreate}>Створити симуляцію</button>
        <button type="button" onClick={onStep} disabled={loading || !canSimulate}>Один крок</button>
        <button type="button" onClick={onRun} disabled={loading || !canSimulate}>Run</button>
        <button type="button" onClick={onReset} disabled={loading}>Скинути все</button>
      </div>
    </div>
  );
}
