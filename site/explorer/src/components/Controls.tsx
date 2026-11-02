import type { ReactNode } from "react";
import type { Tristate } from "../catalog";

export function Chip({
  pressed,
  onClick,
  children,
}: {
  pressed: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      onClick={onClick}
      className={`inline-flex min-h-8 cursor-pointer items-center gap-1.5 rounded-full border px-3 text-[0.8rem] ${
        pressed
          ? "border-brand bg-bg-hover text-fg"
          : "border-line bg-bg text-fg-2 hover:bg-bg-hover"
      }`}
    >
      {children}
    </button>
  );
}

export function Swatch({ color }: { color: string }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block size-2.5 shrink-0 rounded-full"
      style={{ background: color }}
    />
  );
}

const CHOICES: { value: Tristate; label: string }[] = [
  { value: "any", label: "Any" },
  { value: "yes", label: "Yes" },
  { value: "no", label: "No" },
];

export function Choice({
  label,
  value,
  onChange,
  labels = {},
}: {
  label: string;
  value: Tristate;
  onChange: (value: Tristate) => void;
  labels?: Partial<Record<Tristate, string>>;
}) {
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0">
      <legend className="mb-1 p-0 text-[0.75rem] font-semibold text-fg-2">
        {label}
      </legend>
      <div className="inline-flex overflow-hidden rounded-md border border-line">
        {CHOICES.map((choice) => (
          <button
            key={choice.value}
            type="button"
            aria-pressed={value === choice.value}
            onClick={() => onChange(choice.value)}
            className={`min-h-8 cursor-pointer border-0 border-l border-line px-3 text-[0.8rem] first:border-l-0 ${
              value === choice.value
                ? "bg-bg-hover font-semibold text-fg"
                : "bg-bg text-fg-2"
            }`}
          >
            {labels[choice.value] ?? choice.label}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

export function Range({
  label,
  min,
  max,
  onMin,
  onMax,
}: {
  label: string;
  min: string;
  max: string;
  onMin: (value: string) => void;
  onMax: (value: string) => void;
}) {
  const field =
    "min-h-8 w-full min-w-0 rounded-md border border-line bg-bg px-2 text-[0.8rem] tabular-nums";
  return (
    <fieldset className="m-0 min-w-0 border-0 p-0">
      <legend className="mb-1 p-0 text-[0.75rem] font-semibold text-fg-2">
        {label}
      </legend>
      <div className="flex items-center gap-1.5">
        <input
          className={field}
          inputMode="numeric"
          placeholder="min"
          aria-label={`${label}, at least`}
          value={min}
          onChange={(event) => onMin(event.target.value)}
        />
        <span className="text-fg-muted">–</span>
        <input
          className={field}
          inputMode="numeric"
          placeholder="max"
          aria-label={`${label}, at most`}
          value={max}
          onChange={(event) => onMax(event.target.value)}
        />
      </div>
    </fieldset>
  );
}

export function CopyBlock({ code, label }: { code: string; label: string }) {
  return (
    <div className="relative">
      <pre className="m-0 rounded-md border border-line bg-bg-2 p-3 pr-16 font-mono whitespace-pre-wrap wrap-anywhere text-[0.8rem] leading-relaxed">
        <code>{code}</code>
      </pre>
      <button
        type="button"
        aria-label={`Copy ${label}`}
        onClick={(event) => {
          const button = event.currentTarget;
          void navigator.clipboard?.writeText(code).then(() => {
            button.textContent = "Copied";
            setTimeout(() => {
              button.textContent = "Copy";
            }, 1500);
          });
        }}
        className="absolute top-2 right-2 cursor-pointer rounded border border-line bg-bg px-2 py-0.5 text-[0.75rem] text-fg-2 hover:bg-bg-hover"
      >
        Copy
      </button>
    </div>
  );
}
