import { useEffect, useRef, useState } from "react";
import { type Dataset, navigatorUrl } from "../catalog";
import {
  cliSnippet,
  count,
  loadSnippet,
  sourceColor,
  sourceLabel,
  subgraphsSnippet,
  yesNo,
} from "../format";
import { isSearchShortcut } from "../shortcut";
import { Chip, CopyBlock, Swatch } from "./Controls";
import { Description } from "./Description";

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[0.75rem] text-fg-muted">{label}</dt>
      <dd className="m-0 font-semibold tabular-nums">{value}</dd>
    </div>
  );
}

export function Detail({
  dataset,
  mirrored,
  order,
  onClose,
  onSelect,
}: {
  dataset: Dataset | null;
  /** Whether the web mirror holds this network, so the Navigator can open it. */
  mirrored: boolean;
  order: string[];
  onClose: () => void;
  onSelect: (key: string) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  // Keyed by dataset, so opening another one starts from its first subgraph.
  const [picked, setPicked] = useState<{ key: string; name: string }>();

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    if (dataset && !element.open) element.showModal();
    if (!dataset && element.open) element.close();
    element.scrollTop = 0;
  }, [dataset]);

  // On the docs site, Sphinx's sphinx_highlight.js cancels every Escape that
  // reaches the document while focus is off a form control, as it is once a
  // click lands on text in the dialog, and a cancelled Escape leaves a modal
  // open. So the dialog closes itself first, in the window's capture phase.
  // The search shortcut closes it too, since the search behind a modal cannot
  // take focus; App focuses the search when the key bubbles back up.
  const open = dataset !== null;
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" || isSearchShortcut(event))
        dialog.current?.close();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [open]);

  const section = "mx-0 mt-5 mb-1.5 p-0 text-[0.8rem] font-semibold text-fg-2";
  const link = "text-brand underline-offset-2 hover:underline break-all";
  const several = dataset !== null && dataset.subgraph_count > 1;
  const first = dataset?.subgraphs[0];
  const subgraph = picked && picked.key === dataset?.key ? picked.name : first;
  const unlisted = dataset
    ? dataset.subgraph_count - dataset.subgraphs.length
    : 0;
  // furo indents the dd of a classed dl unless it is "simple", and pads headings.
  return (
    // biome-ignore lint/a11y/useKeyWithClickEvents: Escape closes a modal dialog natively; this adds the backdrop click
    <dialog
      ref={dialog}
      aria-labelledby="nx-detail-title"
      onClose={onClose}
      onClick={(event) => {
        // A click on the backdrop lands on the dialog element itself.
        if (event.target === event.currentTarget) onClose();
      }}
      className="m-0 h-dvh max-h-none w-full max-w-none overflow-y-auto border-0 bg-bg p-0 text-fg sm:m-auto sm:h-auto sm:max-h-[min(48rem,90dvh)] sm:w-[min(44rem,92vw)] sm:rounded-xl sm:border sm:border-line sm:shadow-2xl"
    >
      {dataset && (
        <div className="p-4 sm:p-6">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex items-center gap-1.5 text-[0.8rem] text-fg-2">
                <Swatch color={sourceColor(dataset.source, order)} />
                {sourceLabel(dataset.source)}
              </div>
              <h2
                id="nx-detail-title"
                className="m-0 mt-1 p-0 text-xl leading-tight font-bold break-all"
              >
                {dataset.name}
              </h2>
              {dataset.title && dataset.title !== dataset.name && (
                <div className="mt-0.5 text-fg-2">{dataset.title}</div>
              )}
            </div>
            <button
              type="button"
              onClick={onClose}
              aria-label="Close"
              className="min-h-9 min-w-9 shrink-0 cursor-pointer rounded-md border border-line bg-bg text-lg leading-none text-fg-2 hover:bg-bg-hover"
            >
              ×
            </button>
          </div>

          {dataset.domains.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {dataset.domains.map((domain) => (
                <span
                  key={domain}
                  className="rounded-full border border-line px-2.5 py-0.5 text-[0.75rem] text-fg-2"
                >
                  {domain}
                </span>
              ))}
            </div>
          )}

          {dataset.duplicate_of && (
            <p className="mt-4 mb-0 rounded-md border border-line bg-bg-2 p-3 text-[0.85rem]">
              KONECT's copy of a network that cross-source search lists from{" "}
              <button
                type="button"
                className={`cursor-pointer border-0 bg-transparent p-0 font-semibold ${link}`}
                onClick={() => onSelect(dataset.duplicate_of as string)}
              >
                {dataset.duplicate_of}
              </button>
              .
            </p>
          )}

          <dl className="simple m-0 mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
            <Fact label="Nodes" value={count(dataset.nodes)} />
            <Fact label="Edges" value={count(dataset.edges)} />
            <Fact label="Directed" value={yesNo(dataset.directed)} />
            <Fact label="Self-loops" value={yesNo(dataset.loops)} />
            <Fact label="Node metadata" value={yesNo(dataset.node_metadata)} />
          </dl>
          {dataset.node_attributes && dataset.node_attributes.length > 0 && (
            <p className="mt-3 mb-0 text-[0.8rem] text-fg-2">
              Node attributes:{" "}
              {dataset.node_attributes.map((attribute, index) => (
                <span key={attribute}>
                  {index > 0 && ", "}
                  <code className="font-mono">{attribute}</code>
                </span>
              ))}
            </p>
          )}
          {several && (
            <p className="mt-3 mb-0 text-[0.8rem] text-fg-2">
              This name holds {count(dataset.subgraph_count)} networks; the
              figures describe the first,{" "}
              <code className="font-mono">{first}</code>.
            </p>
          )}

          {dataset.description && <Description text={dataset.description} />}

          <h3 className={section}>Load it</h3>
          {several && (
            <fieldset className="m-0 mb-2 min-w-0 border-0 p-0">
              <legend className="mb-1 p-0 text-[0.75rem] font-semibold text-fg-2">
                Subgraph
              </legend>
              <div className="flex flex-wrap gap-1.5">
                {dataset.subgraphs.map((name) => (
                  <Chip
                    key={name}
                    pressed={name === subgraph}
                    onClick={() => setPicked({ key: dataset.key, name })}
                  >
                    <span className="font-mono">{name}</span>
                  </Chip>
                ))}
              </div>
              {unlisted > 0 && (
                <p className="mt-1.5 mb-0 text-[0.8rem] text-fg-2">
                  And {count(unlisted)} more, which{" "}
                  <code className="font-mono">
                    {subgraphsSnippet(dataset.name, dataset.source)}
                  </code>{" "}
                  names.
                </p>
              )}
            </fieldset>
          )}
          <CopyBlock
            code={loadSnippet(
              dataset.name,
              dataset.source,
              subgraph === first ? undefined : subgraph,
            )}
            label="Python code"
          />
          {/* The command line fetches the first subgraph only. */}
          {subgraph === first && (
            <div className="mt-2">
              <CopyBlock
                code={cliSnippet(dataset.name, dataset.source)}
                label="shell command"
              />
            </div>
          )}

          {mirrored && (
            <>
              <h3 className={section}>Explore it</h3>
              <a
                className="inline-flex min-h-9 items-center rounded-md border border-line bg-bg px-3 text-[0.85rem] font-semibold text-brand no-underline hover:bg-bg-hover"
                href={navigatorUrl(dataset.key)}
                target="_blank"
                rel="noopener"
              >
                Open in the Network Navigator ↗
              </a>
              <p className="mt-1.5 mb-0 text-[0.8rem] text-fg-2">
                Finds its modules with Infomap, in your browser.
                {several && (
                  <>
                    {" "}
                    The mirror holds the first network,{" "}
                    <code className="font-mono">{first}</code>.
                  </>
                )}
              </p>
            </>
          )}

          {dataset.citations.length > 0 && (
            <>
              <h3 className={section}>Cite it</h3>
              <ul className="m-0 grid list-none gap-2 p-0">
                {dataset.citations.map((citation) => (
                  <li
                    key={citation.text}
                    className="m-0 rounded-md border border-line bg-bg-2 p-3 text-[0.8rem]"
                  >
                    <div className="break-words whitespace-pre-wrap">
                      {citation.text}
                    </div>
                    {citation.url && (
                      <a
                        className={`mt-1 inline-block ${link}`}
                        href={citation.url}
                      >
                        {citation.url}
                      </a>
                    )}
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3 className={section}>Links</h3>
          <ul className="m-0 grid list-none gap-1 p-0 text-[0.85rem]">
            <li className="m-0">
              <a className={link} href={dataset.url}>
                {sourceLabel(dataset.source)} page
              </a>
            </li>
            {dataset.upstream_url && (
              <li className="m-0">
                <a className={link} href={dataset.upstream_url}>
                  Original publisher
                </a>
              </li>
            )}
          </ul>
        </div>
      )}
    </dialog>
  );
}
