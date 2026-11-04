import type { Filters, Index } from "../catalog";
import { sourceColor, sourceLabel } from "../format";
import { Chip, Choice, Range, Swatch } from "./Controls";

function toggle(values: string[], value: string): string[] {
  return values.includes(value)
    ? values.filter((item) => item !== value)
    : [...values, value];
}

export function FilterPanel({
  index,
  filters,
  attributes,
  mirror,
  onChange,
}: {
  index: Index;
  filters: Filters;
  attributes: string[];
  /** Whether the web mirror's manifest loaded; without it the filter is moot. */
  mirror: boolean;
  onChange: (patch: Partial<Filters>) => void;
}) {
  const order = index.sources.map((source) => source.name);
  const heading = "mb-1.5 text-[0.75rem] font-semibold text-fg-2";
  return (
    <div className="grid gap-4 rounded-lg border border-line bg-bg-2 p-3 sm:p-4">
      <fieldset className="m-0 min-w-0 border-0 p-0">
        <legend className={`p-0 ${heading}`}>Source</legend>
        <div className="flex flex-wrap gap-1.5">
          {index.sources.map((source) => (
            <Chip
              key={source.name}
              pressed={filters.sources.includes(source.name)}
              onClick={() =>
                onChange({ sources: toggle(filters.sources, source.name) })
              }
            >
              <Swatch color={sourceColor(source.name, order)} />
              <span title={source.summary}>{sourceLabel(source.name)}</span>
            </Chip>
          ))}
        </div>
        <label className="mt-2 inline-flex cursor-pointer items-center gap-2 text-[0.8rem] text-fg-2">
          <input
            type="checkbox"
            checked={filters.copies}
            onChange={(event) => onChange({ copies: event.target.checked })}
          />
          Include KONECT's copies of Netzschleuder and SNAP networks
        </label>
      </fieldset>

      <fieldset className="m-0 min-w-0 border-0 p-0">
        <legend className={`p-0 ${heading}`}>Domain</legend>
        <div className="flex flex-wrap gap-1.5">
          {index.domains.map((domain) => (
            <Chip
              key={domain}
              pressed={filters.domains.includes(domain)}
              onClick={() =>
                onChange({ domains: toggle(filters.domains, domain) })
              }
            >
              {domain}
            </Chip>
          ))}
        </div>
      </fieldset>

      <div className="grid grid-cols-1 gap-4 min-[420px]:grid-cols-2 md:grid-cols-4">
        <Range
          label="Nodes"
          min={filters.minNodes}
          max={filters.maxNodes}
          onMin={(minNodes) => onChange({ minNodes })}
          onMax={(maxNodes) => onChange({ maxNodes })}
        />
        <Range
          label="Edges"
          min={filters.minEdges}
          max={filters.maxEdges}
          onMin={(minEdges) => onChange({ minEdges })}
          onMax={(maxEdges) => onChange({ maxEdges })}
        />
        <Choice
          label="Directed"
          value={filters.directed}
          onChange={(directed) => onChange({ directed })}
        />
        <Choice
          label="Self-loops"
          value={filters.loops}
          onChange={(loops) => onChange({ loops })}
        />
      </div>

      <div className="grid grid-cols-1 gap-4 min-[420px]:grid-cols-2 md:grid-cols-4">
        <Choice
          label="Node metadata"
          value={filters.nodeMetadata}
          onChange={(nodeMetadata) => onChange({ nodeMetadata })}
        />
        <label className="block min-w-0 md:col-span-2">
          <span className={`block ${heading}`}>Node attribute</span>
          <input
            list="nx-attributes"
            placeholder="for example gender, year, label"
            value={filters.nodeAttribute}
            onChange={(event) =>
              onChange({ nodeAttribute: event.target.value })
            }
            className="min-h-8 w-full min-w-0 rounded-md border border-line bg-bg px-2 text-[0.8rem]"
          />
          <datalist id="nx-attributes">
            {attributes.map((attribute) => (
              <option key={attribute} value={attribute} />
            ))}
          </datalist>
        </label>
        {mirror && (
          <Choice
            label="In Navigator"
            value={filters.navigator}
            onChange={(navigator) => onChange({ navigator })}
          />
        )}
      </div>
      <p className="m-0 text-[0.75rem] text-fg-muted">
        As in <code>netgamut.search()</code>: a size bound leaves out datasets
        whose source publishes no size, and a yes or no leaves out those no
        source has checked.
      </p>
    </div>
  );
}
