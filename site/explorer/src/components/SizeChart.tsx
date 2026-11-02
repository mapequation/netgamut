import { type MouseEvent, useMemo, useRef, useState } from "react";
import type { Dataset } from "../catalog";
import { compact, count, sourceColor, sourceLabel } from "../format";

const WIDTH = 520;
const HEIGHT = 300;
const PAD = { left: 44, right: 12, top: 10, bottom: 34 };

type Point = { dataset: Dataset; x: number; y: number };

function logScale(low: number, high: number, from: number, to: number) {
  const a = Math.log10(low);
  const b = Math.log10(high);
  return (value: number) =>
    from + ((Math.log10(value) - a) / (b - a)) * (to - from);
}

function decades(low: number, high: number): number[] {
  const ticks = [];
  for (let power = Math.ceil(Math.log10(low)); 10 ** power <= high; power++)
    ticks.push(10 ** power);
  return ticks;
}

/**
 * Nodes against edges on log scales. Every sized dataset is drawn so the axes
 * stay put; those outside the current filters recede to a faint grey.
 */
export function SizeChart({
  all,
  shown,
  order,
  selected,
  onSelect,
}: {
  all: Dataset[];
  shown: Set<string>;
  order: string[];
  selected: string | null;
  onSelect: (key: string) => void;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<Point | null>(null);

  const { points, xTicks, yTicks } = useMemo(() => {
    const sized = all.filter(
      (dataset) => (dataset.nodes ?? 0) > 0 && (dataset.edges ?? 0) > 0,
    ) as (Dataset & { nodes: number; edges: number })[];
    const nodes = sized.map((dataset) => dataset.nodes);
    const edges = sized.map((dataset) => dataset.edges);
    const [x0, x1] = [
      10 ** Math.floor(Math.log10(Math.min(...nodes))),
      Math.max(...nodes) * 1.2,
    ];
    const [y0, y1] = [
      10 ** Math.floor(Math.log10(Math.min(...edges))),
      Math.max(...edges) * 1.2,
    ];
    const x = logScale(x0, x1, PAD.left, WIDTH - PAD.right);
    const y = logScale(y0, y1, HEIGHT - PAD.bottom, PAD.top);
    return {
      points: sized.map((dataset) => ({
        dataset,
        x: x(dataset.nodes),
        y: y(dataset.edges),
      })),
      xTicks: decades(x0, x1).map((value) => ({ value, at: x(value) })),
      yTicks: decades(y0, y1).map((value) => ({ value, at: y(value) })),
    };
  }, [all]);

  const visible = points.filter((point) => shown.has(point.dataset.key));
  const faded = points.filter((point) => !shown.has(point.dataset.key));
  const unsized = shown.size - visible.length;

  function nearest(event: MouseEvent<SVGSVGElement>): Point | null {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return null;
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    const py = ((event.clientY - box.top) / box.height) * HEIGHT;
    let best: Point | null = null;
    let distance = 14 ** 2; // hit radius, larger than the mark
    for (const point of visible) {
      const d = (point.x - px) ** 2 + (point.y - py) ** 2;
      if (d < distance) [best, distance] = [point, d];
    }
    return best;
  }

  const marked = points.find((point) => point.dataset.key === selected);

  return (
    <figure className="m-0 min-w-0">
      <figcaption className="mb-1 text-[0.8rem] font-semibold">
        Size{" "}
        <span className="font-normal text-fg-muted">
          nodes against edges, log scales
          {unsized > 0 &&
            ` · ${unsized.toLocaleString("en")} without published size not drawn`}
        </span>
      </figcaption>
      <div className="relative">
        {/* biome-ignore lint/a11y/useKeyWithClickEvents: the results table is the keyboard path to every point */}
        <svg
          ref={svg}
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          className="block h-auto w-full touch-none select-none"
          role="img"
          aria-label={`Scatter plot of nodes against edges for ${visible.length} datasets. The table below lists the same datasets.`}
          onPointerMove={(event) => setHover(nearest(event))}
          onPointerLeave={() => setHover(null)}
          onClick={(event) => {
            const point = nearest(event);
            if (point) onSelect(point.dataset.key);
          }}
        >
          {xTicks.map((tick) => (
            <g key={`x${tick.value}`}>
              <line
                x1={tick.at}
                x2={tick.at}
                y1={PAD.top}
                y2={HEIGHT - PAD.bottom}
                stroke="var(--color-background-border)"
              />
              <text
                x={tick.at}
                y={HEIGHT - PAD.bottom + 14}
                textAnchor="middle"
                fontSize="10"
                fill="var(--color-foreground-muted)"
              >
                {compact(tick.value)}
              </text>
            </g>
          ))}
          {yTicks.map((tick) => (
            <g key={`y${tick.value}`}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={tick.at}
                y2={tick.at}
                stroke="var(--color-background-border)"
              />
              <text
                x={PAD.left - 6}
                y={tick.at + 3}
                textAnchor="end"
                fontSize="10"
                fill="var(--color-foreground-muted)"
              >
                {compact(tick.value)}
              </text>
            </g>
          ))}
          <text
            x={(PAD.left + WIDTH - PAD.right) / 2}
            y={HEIGHT - 4}
            textAnchor="middle"
            fontSize="10"
            fill="var(--color-foreground-secondary)"
          >
            nodes
          </text>
          <text
            x={10}
            y={(PAD.top + HEIGHT - PAD.bottom) / 2}
            textAnchor="middle"
            fontSize="10"
            fill="var(--color-foreground-secondary)"
            transform={`rotate(-90 10 ${(PAD.top + HEIGHT - PAD.bottom) / 2})`}
          >
            edges
          </text>
          {faded.map((point) => (
            <circle
              key={point.dataset.key}
              cx={point.x}
              cy={point.y}
              r={2.5}
              fill="var(--color-foreground-border)"
              opacity={0.25}
            />
          ))}
          {visible.map((point) => (
            <circle
              key={point.dataset.key}
              cx={point.x}
              cy={point.y}
              r={3.5}
              fill={sourceColor(point.dataset.source, order)}
              stroke="var(--color-background-primary)"
              strokeWidth={1}
            />
          ))}
          {marked && (
            <circle
              cx={marked.x}
              cy={marked.y}
              r={7}
              fill="none"
              stroke="var(--color-foreground-primary)"
              strokeWidth={2}
            />
          )}
          {hover && (
            <circle
              cx={hover.x}
              cy={hover.y}
              r={6}
              fill="none"
              stroke="var(--color-foreground-primary)"
              strokeWidth={1.5}
            />
          )}
        </svg>
        {hover && (
          <div
            className="pointer-events-none absolute z-10 w-max max-w-56 rounded-md border border-line bg-bg px-2.5 py-1.5 text-[0.75rem] shadow-md"
            style={{
              left: `${(hover.x / WIDTH) * 100}%`,
              top: `${(hover.y / HEIGHT) * 100}%`,
              transform: `translate(${hover.x > WIDTH / 2 ? "calc(-100% - 10px)" : "10px"}, -50%)`,
            }}
          >
            <div className="font-semibold break-all text-fg">
              {hover.dataset.name}
            </div>
            <div className="text-fg-2">{sourceLabel(hover.dataset.source)}</div>
            <div className="tabular-nums text-fg-2">
              {count(hover.dataset.nodes)} nodes · {count(hover.dataset.edges)}{" "}
              edges
            </div>
          </div>
        )}
      </div>
    </figure>
  );
}
