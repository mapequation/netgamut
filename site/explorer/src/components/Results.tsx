import type { Dataset, Sort, SortKey } from "../catalog";
import { count, sourceColor, sourceLabel } from "../format";
import { Swatch } from "./Controls";

const COLUMNS: { key: SortKey | null; label: string; numeric?: boolean }[] = [
  { key: "name", label: "Name" },
  { key: "source", label: "Source" },
  { key: null, label: "Domains" },
  { key: "nodes", label: "Nodes", numeric: true },
  { key: "edges", label: "Edges", numeric: true },
  { key: null, label: "Directed" },
];

function direction(dataset: Dataset): string {
  return dataset.directed === null ? "–" : dataset.directed ? "yes" : "no";
}

function SourceTag({ source, order }: { source: string; order: string[] }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
      <Swatch color={sourceColor(source, order)} />
      {sourceLabel(source)}
    </span>
  );
}

export function Results({
  datasets,
  order,
  sort,
  onSort,
  onSelect,
}: {
  datasets: Dataset[];
  order: string[];
  sort: Sort;
  onSort: (sort: Sort) => void;
  onSelect: (key: string) => void;
}) {
  if (datasets.length === 0) {
    return (
      <p className="rounded-lg border border-line p-6 text-center text-fg-2">
        No dataset matches these filters.
      </p>
    );
  }
  return (
    <>
      {/* Phone widths get one card per dataset; the table needs room for six columns. */}
      <ul className="m-0 grid list-none gap-2 p-0 sm:hidden">
        {datasets.map((dataset) => (
          <li key={dataset.key} className="m-0 p-0">
            <button
              type="button"
              onClick={() => onSelect(dataset.key)}
              className="block w-full cursor-pointer rounded-lg border border-line bg-bg p-3 text-left hover:bg-bg-hover"
            >
              <span className="block font-semibold break-all text-brand">
                {dataset.name}
              </span>
              {dataset.title && dataset.title !== dataset.name && (
                <span className="block text-[0.8rem] text-fg-2">
                  {dataset.title}
                </span>
              )}
              <span className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[0.8rem] text-fg-2">
                <SourceTag source={dataset.source} order={order} />
                <span>{dataset.domains.join(", ") || "no domain"}</span>
              </span>
              <span className="mt-0.5 block text-[0.8rem] tabular-nums text-fg-muted">
                {count(dataset.nodes)} nodes · {count(dataset.edges)} edges
                {dataset.directed !== null &&
                  ` · ${dataset.directed ? "directed" : "undirected"}`}
              </span>
            </button>
          </li>
        ))}
      </ul>

      <div className="hidden overflow-x-auto rounded-lg border border-line sm:block">
        <table className="m-0 w-full border-collapse text-[0.8rem]">
          <thead className="bg-bg-2">
            <tr>
              {COLUMNS.map((column) => {
                const active = column.key !== null && sort.key === column.key;
                return (
                  <th
                    key={column.label}
                    scope="col"
                    aria-sort={
                      active
                        ? sort.descending
                          ? "descending"
                          : "ascending"
                        : undefined
                    }
                    className={`border-0 border-b border-line px-3 py-2 font-semibold whitespace-nowrap text-fg-2 ${
                      column.numeric ? "text-right" : "text-left"
                    }`}
                  >
                    {column.key === null ? (
                      column.label
                    ) : (
                      <button
                        type="button"
                        onClick={() =>
                          onSort({
                            key: column.key as SortKey,
                            descending: active
                              ? !sort.descending
                              : !!column.numeric,
                          })
                        }
                        className="inline-flex cursor-pointer items-center gap-1 border-0 bg-transparent p-0 font-semibold text-fg-2 hover:text-fg"
                      >
                        {column.label}
                        <span
                          aria-hidden="true"
                          className={active ? "text-fg" : "text-line-strong"}
                        >
                          {active ? (sort.descending ? "↓" : "↑") : "↕"}
                        </span>
                      </button>
                    )}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {datasets.map((dataset) => (
              <tr
                key={dataset.key}
                onClick={() => onSelect(dataset.key)}
                className="cursor-pointer border-0 border-b border-line last:border-b-0 hover:bg-bg-hover"
              >
                <td className="max-w-64 px-3 py-1.5 align-top">
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(dataset.key);
                    }}
                    className="cursor-pointer border-0 bg-transparent p-0 text-left font-semibold break-all text-brand hover:underline"
                  >
                    {dataset.name}
                  </button>
                  {dataset.title && dataset.title !== dataset.name && (
                    <div
                      className="truncate text-fg-muted"
                      title={dataset.title}
                    >
                      {dataset.title}
                    </div>
                  )}
                </td>
                <td className="px-3 py-1.5 align-top">
                  <SourceTag source={dataset.source} order={order} />
                </td>
                <td className="px-3 py-1.5 align-top text-fg-2">
                  {dataset.domains.join(", ") || "–"}
                </td>
                <td className="px-3 py-1.5 text-right align-top tabular-nums">
                  {count(dataset.nodes)}
                </td>
                <td className="px-3 py-1.5 text-right align-top tabular-nums">
                  {count(dataset.edges)}
                </td>
                <td className="px-3 py-1.5 align-top text-fg-2">
                  {direction(dataset)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
