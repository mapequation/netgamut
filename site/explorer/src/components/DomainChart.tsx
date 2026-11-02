import type { Dataset } from "../catalog";

/**
 * Datasets per domain: the current selection over all of *all*, one bar per
 * domain in netgamut's own order. A bar toggles that domain's filter.
 */
export function DomainChart({
  domains,
  all,
  shown,
  active,
  onToggle,
}: {
  domains: string[];
  all: Dataset[];
  shown: Dataset[];
  active: string[];
  onToggle: (domain: string) => void;
}) {
  const tally = (datasets: Dataset[]) => {
    const counts = new Map(domains.map((domain) => [domain, 0]));
    for (const dataset of datasets)
      for (const domain of dataset.domains)
        counts.set(domain, (counts.get(domain) ?? 0) + 1);
    return counts;
  };
  const totals = tally(all);
  const counts = tally(shown);
  const scale = Math.max(1, ...totals.values());

  return (
    <figure className="m-0 min-w-0">
      <figcaption className="mb-1 text-[0.8rem] font-semibold">
        Domains{" "}
        <span className="font-normal text-fg-muted">
          selected of all datasets
        </span>
      </figcaption>
      <div className="grid gap-0.5">
        {domains.map((domain) => {
          const value = counts.get(domain) ?? 0;
          const total = totals.get(domain) ?? 0;
          return (
            <button
              key={domain}
              type="button"
              aria-pressed={active.includes(domain)}
              title={`${domain}: ${value} of ${total} datasets`}
              onClick={() => onToggle(domain)}
              className={`grid cursor-pointer grid-cols-[7.5rem_1fr_4.5rem] items-center gap-2 rounded border-0 bg-transparent px-1 py-0.5 text-left text-[0.75rem] hover:bg-bg-hover ${
                active.includes(domain) ? "font-semibold text-fg" : "text-fg-2"
              }`}
            >
              <span className="truncate">{domain}</span>
              <span className="relative block h-2.5">
                <span
                  className="absolute inset-y-0 left-0 rounded-r-sm bg-line"
                  style={{ width: `${(total / scale) * 100}%` }}
                />
                <span
                  className="absolute inset-y-0 left-0 rounded-r-sm"
                  style={{
                    width: `${(value / scale) * 100}%`,
                    background: "var(--color-brand-content)",
                  }}
                />
              </span>
              <span className="text-right tabular-nums">
                {value}
                <span className="text-fg-muted"> / {total}</span>
              </span>
            </button>
          );
        })}
      </div>
    </figure>
  );
}
