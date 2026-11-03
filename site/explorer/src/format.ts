// Display names for the source keys the index uses; an unknown key shows as is.
const LABELS: Record<string, string> = {
  snap: "SNAP",
  netzschleuder: "Netzschleuder",
  konect: "KONECT",
  ogb: "OGB",
  pyg: "PyG",
  paths: "paths",
  ahorn: "AHORN",
};

export function sourceLabel(source: string): string {
  return LABELS[source] ?? source;
}

/** The categorical slot of a source: its position in the index, never its rank. */
export function sourceColor(source: string, order: string[]): string {
  const slot = order.indexOf(source);
  return slot >= 0 && slot < 8
    ? `var(--series-${slot + 1})`
    : "var(--series-other)";
}

export function count(value: number | null): string {
  return value === null ? "–" : value.toLocaleString("en");
}

export function compact(value: number): string {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function yesNo(value: boolean | null): string {
  return value === null ? "unknown" : value ? "yes" : "no";
}

/** The Python call that loads a dataset, with the name as a string literal. */
export function loadSnippet(
  name: string,
  source: string,
  subgraph?: string,
): string {
  const pick = subgraph ? `, subgraph=${JSON.stringify(subgraph)}` : "";
  return `import netgamut\n\nnet = netgamut.load(${JSON.stringify(name)}, source=${JSON.stringify(source)}${pick})`;
}

/** The Python call that names every subgraph, past the index's preview. */
export function subgraphsSnippet(name: string, source: string): string {
  return `netgamut.${source}.info(${JSON.stringify(name)}).subgraph_names()`;
}

export function cliSnippet(name: string, source: string): string {
  const quoted = /^[\w.+-]+$/.test(name)
    ? name
    : `'${name.replaceAll("'", "'\\''")}'`;
  return `netgamut get ${quoted} --source ${source}`;
}
