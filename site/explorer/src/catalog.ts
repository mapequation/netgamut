// The shape `netgamut export-index` writes: one `Match.to_dict()` per dataset.
// Every cross-source decision (domains, KONECT's duplicates, loops) is made in
// Python; this file only filters and sorts what the index already says.

export interface Citation {
  text: string;
  url: string;
}

export interface Dataset {
  key: string;
  source: string;
  name: string;
  title: string;
  domains: string[];
  nodes: number | null;
  edges: number | null;
  directed: boolean | null;
  loops: boolean | null;
  node_metadata: boolean | null;
  node_attributes: string[] | null;
  /** How many networks the name holds; the figures describe the first. */
  subgraph_count: number;
  /** The first few of them, by name; empty where the source names none. */
  subgraphs: string[];
  description: string;
  citations: Citation[];
  url: string;
  upstream_url: string;
  duplicate_of: string | null;
}

export interface Source {
  name: string;
  summary: string;
  retrieved: string;
}

export interface Index {
  netgamut: string;
  domains: string[];
  sources: Source[];
  datasets: Dataset[];
}

/** The web mirror's manifest (`netgamut export-networks`): networks a browser can load. */
export interface Mirror {
  datasets: { key: string }[];
}

const NAVIGATOR = "https://mapequation.org/navigator/";

/** Opens a mirrored network in the Network Navigator, which runs Infomap on it. */
export function navigatorUrl(key: string): string {
  return `${NAVIGATOR}?netgamut=${encodeURIComponent(key).replace("%2F", "/")}`;
}

/** Yes / no / either, for the boolean fields where the index may say "unknown". */
export type Tristate = "any" | "yes" | "no";

export interface Filters {
  query: string;
  sources: string[];
  domains: string[];
  directed: Tristate;
  loops: Tristate;
  nodeMetadata: Tristate;
  nodeAttribute: string;
  minNodes: string;
  maxNodes: string;
  minEdges: string;
  maxEdges: string;
  copies: boolean;
}

export const NO_FILTERS: Filters = {
  query: "",
  sources: [],
  domains: [],
  directed: "any",
  loops: "any",
  nodeMetadata: "any",
  nodeAttribute: "",
  minNodes: "",
  maxNodes: "",
  minEdges: "",
  maxEdges: "",
  copies: false,
};

export type SortKey = "name" | "source" | "nodes" | "edges";

export interface Sort {
  key: SortKey;
  descending: boolean;
}

function bound(text: string): number | null {
  const value = Number(text.replace(/[\s,_]/g, ""));
  return text.trim() === "" || Number.isNaN(value) ? null : value;
}

// Like search(): with a bound set, a dataset whose size is unpublished is out.
function within(
  value: number | null,
  low: number | null,
  high: number | null,
): boolean {
  if (low === null && high === null) return true;
  if (value === null) return false;
  return (low === null || value >= low) && (high === null || value <= high);
}

// Like search(): "unknown" never matches a yes or a no.
function matches(value: boolean | null, wanted: Tristate): boolean {
  return wanted === "any" || value === (wanted === "yes");
}

export function filterDatasets(
  datasets: Dataset[],
  filters: Filters,
): Dataset[] {
  const needle = filters.query.trim().toLocaleLowerCase();
  const attribute = filters.nodeAttribute.trim().toLocaleLowerCase();
  const nodes = [bound(filters.minNodes), bound(filters.maxNodes)] as const;
  const edges = [bound(filters.minEdges), bound(filters.maxEdges)] as const;
  return datasets.filter(
    (dataset) =>
      (filters.copies || dataset.duplicate_of === null) &&
      (filters.sources.length === 0 ||
        filters.sources.includes(dataset.source)) &&
      (filters.domains.length === 0 ||
        dataset.domains.some((domain) => filters.domains.includes(domain))) &&
      (needle === "" ||
        `${dataset.name} ${dataset.title} ${dataset.description}`
          .toLocaleLowerCase()
          .includes(needle)) &&
      matches(dataset.directed, filters.directed) &&
      matches(dataset.loops, filters.loops) &&
      matches(dataset.node_metadata, filters.nodeMetadata) &&
      (attribute === "" ||
        (dataset.node_attributes ?? []).some(
          (column) => column.toLocaleLowerCase() === attribute,
        )) &&
      within(dataset.nodes, ...nodes) &&
      within(dataset.edges, ...edges),
  );
}

export function sortDatasets(
  datasets: Dataset[],
  sort: Sort,
  order: string[],
): Dataset[] {
  const sign = sort.descending ? -1 : 1;
  const compare = (a: Dataset, b: Dataset): number => {
    if (sort.key === "name") return sign * a.name.localeCompare(b.name, "en");
    if (sort.key === "source")
      return sign * (order.indexOf(a.source) - order.indexOf(b.source));
    const x = a[sort.key];
    const y = b[sort.key];
    // Unpublished sizes go last whichever way the column is sorted.
    if (x === null || y === null)
      return Number(x === null) - Number(y === null);
    return sign * (x - y);
  };
  return [...datasets].sort(
    (a, b) => compare(a, b) || a.key.localeCompare(b.key),
  );
}

export function countFilters(filters: Filters): number {
  return (Object.keys(NO_FILTERS) as (keyof Filters)[]).filter((key) => {
    const value = filters[key];
    return Array.isArray(value) ? value.length > 0 : value !== NO_FILTERS[key];
  }).length;
}
