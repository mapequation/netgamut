import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  countFilters,
  type Filters,
  filterDatasets,
  type Index,
  type Mirror,
  NO_FILTERS,
  type Sort,
  sortDatasets,
} from "./catalog";
import { Detail } from "./components/Detail";
import { DomainChart } from "./components/DomainChart";
import { FilterPanel } from "./components/FilterPanel";
import { Results } from "./components/Results";
import { SizeChart } from "./components/SizeChart";
import { isSearchShortcut, SEARCH_ARIA, SEARCH_KEYS } from "./shortcut";

const PAGE = 50;
const HASH = "#dataset=";

function hashKey(): string | null {
  return window.location.hash.startsWith(HASH)
    ? decodeURIComponent(window.location.hash.slice(HASH.length))
    : null;
}

const wide = () => window.matchMedia("(min-width: 640px)").matches;

export function App({
  indexUrl,
  mirrorUrl,
}: {
  indexUrl: string;
  mirrorUrl: string;
}) {
  const [index, setIndex] = useState<Index | null>(null);
  const [mirrored, setMirrored] = useState<ReadonlySet<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<Filters>(NO_FILTERS);
  const [sort, setSort] = useState<Sort>({ key: "name", descending: false });
  const [limit, setLimit] = useState(PAGE);
  const [selected, setSelected] = useState<string | null>(hashKey);
  const [showFilters, setShowFilters] = useState(wide);
  const [showCharts, setShowCharts] = useState(wide);
  const deferred = useDeferredValue(filters);
  const search = useRef<HTMLInputElement>(null);

  // In the bubble phase: an open dataset's dialog closes on the same key in the
  // capture phase (Detail), so the search is no longer inert by now.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!isSearchShortcut(event)) return;
      event.preventDefault();
      search.current?.focus();
      search.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    fetch(indexUrl)
      .then((response) => {
        if (!response.ok)
          throw new Error(`${response.status} ${response.statusText}`);
        return response.json() as Promise<Index>;
      })
      .then(setIndex)
      .catch((reason: unknown) => setError(String(reason)));
  }, [indexUrl]);

  // Optional: without the mirror the explorer works as before, minus the
  // Navigator links.
  useEffect(() => {
    fetch(mirrorUrl)
      .then((response) =>
        response.ok ? (response.json() as Promise<Mirror>) : null,
      )
      .then((mirror) => {
        if (mirror)
          setMirrored(new Set(mirror.datasets.map((entry) => entry.key)));
      })
      .catch(() => {});
  }, [mirrorUrl]);

  useEffect(() => {
    const follow = () => setSelected(hashKey());
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

  // The open dataset lives in the URL, so a link can point at one.
  const select = useCallback((key: string | null) => {
    const { pathname, search } = window.location;
    const hash = key === null ? "" : `${HASH}${encodeURIComponent(key)}`;
    window.history.replaceState(null, "", `${pathname}${search}${hash}`);
    setSelected(key);
  }, []);

  const order = useMemo(
    () => index?.sources.map((source) => source.name) ?? [],
    [index],
  );
  const base = useMemo(
    () =>
      index?.datasets.filter(
        (dataset) => deferred.copies || dataset.duplicate_of === null,
      ) ?? [],
    [index, deferred.copies],
  );
  const filtered = useMemo(
    () => (index ? filterDatasets(index.datasets, deferred, mirrored) : []),
    [index, deferred, mirrored],
  );
  const sorted = useMemo(
    () => sortDatasets(filtered, sort, order),
    [filtered, sort, order],
  );
  const shown = useMemo(
    () => new Set(filtered.map((dataset) => dataset.key)),
    [filtered],
  );
  const attributes = useMemo(() => {
    const names = new Set<string>();
    for (const dataset of index?.datasets ?? [])
      for (const name of dataset.node_attributes ?? [])
        names.add(name.toLocaleLowerCase());
    return [...names].sort();
  }, [index]);

  if (error) {
    return (
      <p className="rounded-lg border border-line p-4">
        Could not load the catalog index ({error}). <code>netgamut list</code>{" "}
        prints the same catalog.
      </p>
    );
  }
  if (!index) return <p className="text-fg-2">Loading the catalog…</p>;

  const change = (patch: Partial<Filters>) => {
    setFilters((current) => ({ ...current, ...patch }));
    setLimit(PAGE);
  };
  const active = countFilters(filters);
  const current =
    index.datasets.find((dataset) => dataset.key === selected) ?? null;
  const toggle =
    "inline-flex min-h-9 cursor-pointer items-center gap-1.5 rounded-md border border-line bg-bg px-3 text-[0.8rem] text-fg-2 hover:bg-bg-hover";

  return (
    <div className="grid gap-4 text-[0.9rem] text-fg">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative flex min-w-0 flex-[1_1_16rem] items-center">
          <input
            ref={search}
            type="search"
            value={filters.query}
            onChange={(event) => change({ query: event.target.value })}
            placeholder={`Search ${base.length.toLocaleString("en")} datasets`}
            aria-label="Search datasets"
            aria-keyshortcuts={SEARCH_ARIA}
            className={`min-h-9 w-full min-w-0 rounded-md border border-line-strong bg-bg pl-3 ${
              filters.query === "" ? "pr-14" : "pr-3"
            }`}
          />
          {/* A span, not a kbd, which furo styles; the input names the
              shortcut to assistive technology. Gone once there is a query,
              where the browser draws its clear button. */}
          {filters.query === "" && (
            <span
              aria-hidden="true"
              className="pointer-events-none absolute right-2 rounded border border-line bg-bg-2 px-1.5 text-[0.7rem] leading-5 text-fg-muted"
            >
              {SEARCH_KEYS}
            </span>
          )}
        </div>
        <button
          type="button"
          aria-expanded={showFilters}
          onClick={() => setShowFilters(!showFilters)}
          className={toggle}
        >
          Filters{active > 0 && ` (${active})`}
        </button>
        <button
          type="button"
          aria-expanded={showCharts}
          onClick={() => setShowCharts(!showCharts)}
          className={toggle}
        >
          Charts
        </button>
        {active > 0 && (
          <button
            type="button"
            onClick={() => change(NO_FILTERS)}
            className={toggle}
          >
            Clear
          </button>
        )}
      </div>

      {showFilters && (
        <FilterPanel
          index={index}
          filters={filters}
          attributes={attributes}
          mirror={mirrored.size > 0}
          onChange={change}
        />
      )}

      {showCharts && (
        <div className="grid gap-6 md:grid-cols-[3fr_2fr]">
          <SizeChart
            all={base}
            shown={shown}
            order={order}
            selected={selected}
            onSelect={select}
          />
          <DomainChart
            domains={index.domains}
            all={base}
            shown={filtered}
            active={filters.domains}
            onToggle={(domain) =>
              change({
                domains: filters.domains.includes(domain)
                  ? filters.domains.filter((item) => item !== domain)
                  : [...filters.domains, domain],
              })
            }
          />
        </div>
      )}

      <div className="flex flex-wrap items-baseline justify-between gap-2 text-[0.8rem] text-fg-2">
        <span aria-live="polite">
          <strong className="text-fg">
            {filtered.length.toLocaleString("en")}
          </strong>{" "}
          of {base.length.toLocaleString("en")} datasets
        </span>
        <span className="text-fg-muted">
          netgamut {index.netgamut} · catalogs retrieved{" "}
          {index.sources
            .map((source) => source.retrieved.slice(0, 10))
            .sort()
            .at(0)}{" "}
          to{" "}
          {index.sources
            .map((source) => source.retrieved.slice(0, 10))
            .sort()
            .at(-1)}
        </span>
      </div>

      <Results
        datasets={sorted.slice(0, limit)}
        order={order}
        sort={sort}
        onSort={setSort}
        onSelect={select}
      />
      {sorted.length > limit && (
        <div className="flex justify-center gap-2">
          <button
            type="button"
            onClick={() => setLimit(limit + PAGE)}
            className={toggle}
          >
            Show {Math.min(PAGE, sorted.length - limit)} more
          </button>
          <button
            type="button"
            onClick={() => setLimit(sorted.length)}
            className={toggle}
          >
            Show all {sorted.length.toLocaleString("en")}
          </button>
        </div>
      )}

      <Detail
        dataset={current}
        mirrored={current !== null && mirrored.has(current.key)}
        order={order}
        onClose={() => select(null)}
        onSelect={select}
      />
    </div>
  );
}
