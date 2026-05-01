import { useEffect, useRef, useState } from "react";
import type { PointDto } from "../types/simulation";
import {
  DEFAULT_VIEWPORT,
  drawAxes,
  fitViewportToPoints,
  screenToWorld,
  worldToScreen,
  zoomAt,
  type Viewport,
} from "../utils/canvasViewport";

type DrawingCanvasProps = {
  points: PointDto[];
  onChange: (points: PointDto[]) => void;
  closed: boolean;
  width?: number;
  height?: number;
};

function findPointIndex(points: PointDto[], target: PointDto, radius = 10): number {
  return points.findIndex((point) => {
    const dx = point.x - target.x;
    const dy = point.y - target.y;
    return Math.sqrt(dx * dx + dy * dy) <= radius;
  });
}

export function DrawingCanvas({
  points,
  onChange,
  closed,
  width = 760,
  height = 560,
}: DrawingCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverPoint, setHoverPoint] = useState<PointDto | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const [viewport, setViewport] = useState<Viewport>(DEFAULT_VIEWPORT);
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef<{ x: number; y: number; offsetX: number; offsetY: number } | null>(null);

  useEffect(() => {
    if (points.length === 0) {
      setViewport(DEFAULT_VIEWPORT);
      return;
    }
    setViewport((current) => (current === DEFAULT_VIEWPORT ? fitViewportToPoints(points, width, height) : current));
  }, [points.length, width, height]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#06111f";
    ctx.fillRect(0, 0, width, height);

    drawAxes(ctx, width, height, viewport);

    if (points.length > 0) {
      const first = worldToScreen(points[0], viewport, height);
      ctx.beginPath();
      ctx.moveTo(first.x, first.y);

      for (let i = 1; i < points.length; i++) {
        const screenPoint = worldToScreen(points[i], viewport, height);
        ctx.lineTo(screenPoint.x, screenPoint.y);
      }

      if (closed && points.length >= 3) {
        ctx.closePath();
      } else if (hoverPoint) {
        const hoverScreen = worldToScreen(hoverPoint, viewport, height);
        ctx.lineTo(hoverScreen.x, hoverScreen.y);
      }

      ctx.strokeStyle = "#5ab2ff";
      ctx.lineWidth = 2;
      ctx.stroke();

      if (closed && points.length >= 3) {
        ctx.fillStyle = "rgba(90,178,255,0.10)";
        ctx.fill();
      }
    }

    for (let i = 0; i < points.length; i++) {
      const point = points[i];
      const screenPoint = worldToScreen(point, viewport, height);
      ctx.beginPath();
      ctx.arc(screenPoint.x, screenPoint.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();

      ctx.fillStyle = "#cbd5e1";
      ctx.font = "12px sans-serif";
      ctx.fillText(`${i} (${point.x.toFixed(1)}, ${point.y.toFixed(1)})`, screenPoint.x + 8, screenPoint.y - 8);
    }

    ctx.fillStyle = "rgba(226,232,240,0.85)";
    ctx.font = "12px sans-serif";
    ctx.fillText("Колесо миші — зум, Shift + drag — панорамування, double click — fit", 16, 20);
  }, [points, hoverPoint, closed, width, height, viewport]);

  function getMousePosition(event: React.MouseEvent<HTMLCanvasElement>): PointDto {
    const rect = event.currentTarget.getBoundingClientRect();
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  function handleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    if (closed || isPanning) return;

    const mouse = getMousePosition(event);
    const worldMouse = screenToWorld(mouse, viewport, height);
    const existingIndex = findPointIndex(points, worldMouse, 8 / viewport.scale);

    if (existingIndex !== -1) return;
    onChange([...points, worldMouse]);
  }

  function handleMouseMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const mouse = getMousePosition(event);

    if (isPanning && panStartRef.current) {
      const dx = mouse.x - panStartRef.current.x;
      const dy = mouse.y - panStartRef.current.y;
      setViewport({
        scale: viewport.scale,
        offsetX: panStartRef.current.offsetX + dx,
        offsetY: panStartRef.current.offsetY - dy,
      });
      return;
    }

    const worldMouse = screenToWorld(mouse, viewport, height);

    if (dragIndex !== null) {
      const next = [...points];
      next[dragIndex] = worldMouse;
      onChange(next);
      return;
    }

    if (!closed) {
      setHoverPoint(worldMouse);
    }
  }

  function handleMouseDown(event: React.MouseEvent<HTMLCanvasElement>) {
    const mouse = getMousePosition(event);

    if (event.shiftKey || event.button === 1) {
      setIsPanning(true);
      panStartRef.current = { x: mouse.x, y: mouse.y, offsetX: viewport.offsetX, offsetY: viewport.offsetY };
      return;
    }

    const worldMouse = screenToWorld(mouse, viewport, height);
    const index = findPointIndex(points, worldMouse, 10 / viewport.scale);
    if (index !== -1) {
      setDragIndex(index);
    }
  }

  function handleMouseUp() {
    setDragIndex(null);
    setIsPanning(false);
    panStartRef.current = null;
  }

  function handleMouseLeave() {
    setDragIndex(null);
    setHoverPoint(null);
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
    setViewport(points.length > 0 ? fitViewportToPoints(points, width, height) : DEFAULT_VIEWPORT);
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      onClick={handleClick}
      onMouseMove={handleMouseMove}
      onMouseDown={handleMouseDown}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseLeave}
      onWheel={handleWheel}
      onDoubleClick={handleDoubleClick}
      style={{
        width: "100%",
        maxWidth: width,
        borderRadius: 18,
        border: "1px solid rgba(255,255,255,0.10)",
        display: "block",
        cursor: isPanning ? "grabbing" : dragIndex !== null ? "grabbing" : "crosshair",
        background: "#06111f",
      }}
    />
  );
}
