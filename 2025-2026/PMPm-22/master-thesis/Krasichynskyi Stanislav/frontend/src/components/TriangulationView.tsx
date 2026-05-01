import { useEffect, useMemo, useRef, useState } from "react";
import type { PointDto, SimulationStateDto } from "../types/simulation";
import {
  DEFAULT_VIEWPORT,
  drawAxes,
  fitViewportToPoints,
  worldToScreen,
  zoomAt,
  type Viewport,
} from "../utils/canvasViewport";

type ResultField = "concentration" | "pressure";

type TriangulationViewProps = {
  state: SimulationStateDto | null;
  width?: number;
  height?: number;
  showValues?: boolean;
  field: ResultField;
};

function normalize(value: number, min: number, max: number): number {
  if (max === min) return 0.5;
  return (value - min) / (max - min);
}

function getHeatColor(value: number, min: number, max: number): string {
  const t = normalize(value, min, max);
  const r = Math.round(255 * t);
  const g = Math.round(200 * (1 - Math.abs(t - 0.5) * 2));
  const b = Math.round(255 * (1 - t));
  return `rgb(${r}, ${g}, ${b})`;
}

function drawPolygon(ctx: CanvasRenderingContext2D, points: PointDto[], viewport: Viewport, height: number) {
  if (points.length < 2) return;

  const first = worldToScreen(points[0], viewport, height);
  ctx.beginPath();
  ctx.moveTo(first.x, first.y);

  for (let i = 1; i < points.length; i++) {
    const p = worldToScreen(points[i], viewport, height);
    ctx.lineTo(p.x, p.y);
  }

  ctx.closePath();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();
}

export function TriangulationView({
  state,
  width = 760,
  height = 560,
  showValues = false,
  field,
}: TriangulationViewProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [viewport, setViewport] = useState<Viewport>(DEFAULT_VIEWPORT);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);

  const values = field === "pressure" ? (state?.pressure ?? []) : (state?.concentration ?? []);

  const stats = useMemo(() => {
    if (!values.length) return { min: 0, max: 1 };
    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [values]);

  useEffect(() => {
    if (!state || state.vertices.length === 0) {
      setViewport(DEFAULT_VIEWPORT);
      return;
    }
    setViewport(fitViewportToPoints(state.vertices, width, height));
  }, [state, width, height, field]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, width, height);
    drawAxes(ctx, width, height, viewport);

    if (!state) return;

    const { vertices, triangles, points } = state;

    for (const triangle of triangles) {
      const p1 = vertices[triangle.a];
      const p2 = vertices[triangle.b];
      const p3 = vertices[triangle.c];

      if (!p1 || !p2 || !p3) continue;

      const avgValue =
        ((values[triangle.a] ?? 0) + (values[triangle.b] ?? 0) + (values[triangle.c] ?? 0)) / 3;

      const s1 = worldToScreen(p1, viewport, height);
      const s2 = worldToScreen(p2, viewport, height);
      const s3 = worldToScreen(p3, viewport, height);

      ctx.beginPath();
      ctx.moveTo(s1.x, s1.y);
      ctx.lineTo(s2.x, s2.y);
      ctx.lineTo(s3.x, s3.y);
      ctx.closePath();

      ctx.fillStyle = getHeatColor(avgValue, stats.min, stats.max);
      ctx.fill();

      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    drawPolygon(ctx, points, viewport, height);

    for (let i = 0; i < vertices.length; i++) {
      const vertex = vertices[i];
      const value = values[i] ?? 0;
      const screenPoint = worldToScreen(vertex, viewport, height);

      ctx.beginPath();
      ctx.arc(screenPoint.x, screenPoint.y, 2.8, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      if (showValues) {
        ctx.fillStyle = "#e2e8f0";
        ctx.font = "11px sans-serif";
        ctx.fillText(value.toFixed(2), screenPoint.x + 5, screenPoint.y - 5);
      }
    }

    ctx.fillStyle = "rgba(226,232,240,0.85)";
    ctx.font = "12px sans-serif";
    ctx.fillText("Колесо миші — зум, Shift + drag — панорамування, double click — fit", 16, 20);
  }, [state, width, height, showValues, viewport, field, values, stats]);

  function getMousePosition(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    const mouse = getMousePosition(event);
    if (event.shiftKey || event.button === 0 || event.button === 1) {
      setIsPanning(true);
      panStartRef.current = { x: mouse.x, y: mouse.y, offsetX: viewport.offsetX, offsetY: viewport.offsetY };
    }
  }

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    if (!isPanning || !panStartRef.current) return;
    const mouse = getMousePosition(event);
    const dx = mouse.x - panStartRef.current.x;
    const dy = mouse.y - panStartRef.current.y;
    setViewport({
      scale: viewport.scale,
      offsetX: panStartRef.current.offsetX + dx,
      offsetY: panStartRef.current.offsetY - dy,
    });
  }

  function handleMouseUp() {
    setIsPanning(false);
    panStartRef.current = null;
  }

  function handleWheel(event: React.WheelEvent<HTMLCanvasElement>) {
    event.preventDefault();
    const mouse = getMousePosition(event);
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
    setViewport((current) => zoomAt(current, mouse, factor, height));
  }

  function handleDoubleClick() {
    setViewport(state?.vertices?.length ? fitViewportToPoints(state.vertices, width, height) : DEFAULT_VIEWPORT);
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
      style={{
        width: "100%",
        maxWidth: width,
        borderRadius: 18,
        border: "1px solid rgba(255,255,255,0.10)",
        display: "block",
        background: "#020617",
        cursor: isPanning ? "grabbing" : "grab",
      }}
    />
  );
}
