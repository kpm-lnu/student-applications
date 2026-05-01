export type PointLike = { x: number; y: number };

export type Viewport = {
  scale: number;
  offsetX: number;
  offsetY: number;
};

export const DEFAULT_VIEWPORT: Viewport = {
  scale: 1,
  offsetX: 40,
  offsetY: 40,
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function worldToScreen(point: PointLike, viewport: Viewport, height: number): PointLike {
  return {
    x: point.x * viewport.scale + viewport.offsetX,
    y: height - (point.y * viewport.scale + viewport.offsetY),
  };
}

export function screenToWorld(point: PointLike, viewport: Viewport, height: number): PointLike {
  return {
    x: (point.x - viewport.offsetX) / viewport.scale,
    y: ((height - point.y) - viewport.offsetY) / viewport.scale,
  };
}

export function zoomAt(
  viewport: Viewport,
  screenPoint: PointLike,
  zoomFactor: number,
  height: number,
): Viewport {
  const worldBefore = screenToWorld(screenPoint, viewport, height);
  const nextScale = clamp(viewport.scale * zoomFactor, 0.2, 20);
  const nextViewport = {
    scale: nextScale,
    offsetX: screenPoint.x - worldBefore.x * nextScale,
    offsetY: height - screenPoint.y - worldBefore.y * nextScale,
  };
  return nextViewport;
}

export function fitViewportToPoints(
  points: PointLike[],
  width: number,
  height: number,
  padding = 50,
): Viewport {
  if (points.length === 0) {
    return { ...DEFAULT_VIEWPORT };
  }

  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  const minY = Math.min(...points.map((p) => p.y));
  const maxY = Math.max(...points.map((p) => p.y));

  const worldWidth = Math.max(maxX - minX, 1);
  const worldHeight = Math.max(maxY - minY, 1);

  const scaleX = (width - padding * 2) / worldWidth;
  const scaleY = (height - padding * 2) / worldHeight;
  const scale = clamp(Math.min(scaleX, scaleY), 0.2, 20);

  const usedWidth = worldWidth * scale;
  const usedHeight = worldHeight * scale;

  const offsetX = padding + (width - padding * 2 - usedWidth) / 2 - minX * scale;
  const offsetY = padding + (height - padding * 2 - usedHeight) / 2 - minY * scale;

  return {
    scale,
    offsetX,
    offsetY,
  };
}

export function getNiceStep(rawStep: number): number {
  if (rawStep <= 0 || !Number.isFinite(rawStep)) return 1;
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / power;

  if (normalized <= 1) return 1 * power;
  if (normalized <= 2) return 2 * power;
  if (normalized <= 5) return 5 * power;
  return 10 * power;
}

export function drawAxes(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  viewport: Viewport,
) {
  const xWorldMin = screenToWorld({ x: 0, y: height }, viewport, height).x;
  const xWorldMax = screenToWorld({ x: width, y: height }, viewport, height).x;
  const yWorldMin = screenToWorld({ x: 0, y: height }, viewport, height).y;
  const yWorldMax = screenToWorld({ x: 0, y: 0 }, viewport, height).y;

  const xStep = getNiceStep((xWorldMax - xWorldMin) / 10);
  const yStep = getNiceStep((yWorldMax - yWorldMin) / 10);

  ctx.save();
  ctx.font = "11px sans-serif";

  for (let x = Math.floor(xWorldMin / xStep) * xStep; x <= xWorldMax; x += xStep) {
    const sx = worldToScreen({ x, y: 0 }, viewport, height).x;
    ctx.beginPath();
    ctx.moveTo(sx, 0);
    ctx.lineTo(sx, height);
    ctx.strokeStyle = Math.abs(x) < xStep / 2 ? "rgba(148,163,184,0.55)" : "rgba(255,255,255,0.06)";
    ctx.lineWidth = Math.abs(x) < xStep / 2 ? 1.5 : 1;
    ctx.stroke();

    if (sx >= 0 && sx <= width) {
      ctx.fillStyle = "rgba(226,232,240,0.85)";
      ctx.fillText(Number(x.toFixed(2)).toString(), sx + 4, height - 8);
    }
  }

  for (let y = Math.floor(yWorldMin / yStep) * yStep; y <= yWorldMax; y += yStep) {
    const sy = worldToScreen({ x: 0, y }, viewport, height).y;
    ctx.beginPath();
    ctx.moveTo(0, sy);
    ctx.lineTo(width, sy);
    ctx.strokeStyle = Math.abs(y) < yStep / 2 ? "rgba(148,163,184,0.55)" : "rgba(255,255,255,0.06)";
    ctx.lineWidth = Math.abs(y) < yStep / 2 ? 1.5 : 1;
    ctx.stroke();

    if (sy >= 0 && sy <= height) {
      ctx.fillStyle = "rgba(226,232,240,0.85)";
      ctx.fillText(Number(y.toFixed(2)).toString(), 8, sy - 4);
    }
  }

  ctx.fillStyle = "rgba(226,232,240,0.9)";
  ctx.fillText("X", width - 18, height - 12);
  ctx.fillText("Y", 10, 16);
  ctx.restore();
}
