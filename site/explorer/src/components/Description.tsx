import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const link = "text-brand underline-offset-2 hover:underline break-all";

// Descriptions are Markdown: Netzschleuder's hold lists, autolinks and a
// footnote crediting ICON; AHORN's, inline code. Raw HTML is not rendered.
const components: Components = {
  p: ({ node, ...props }) => <p {...props} className="m-0 mt-2 first:mt-0" />,
  ul: ({ node, ...props }) => <ul {...props} className="m-0 mt-2 pl-5" />,
  ol: ({ node, ...props }) => <ol {...props} className="m-0 mt-2 pl-5" />,
  li: ({ node, ...props }) => <li {...props} className="m-0 mt-1" />,
  code: ({ node, ...props }) => (
    <code {...props} className="font-mono text-[0.9em]" />
  ),
  a: ({ node, href = "", ...props }) => (
    <a
      {...props}
      href={href}
      className={link}
      onClick={(event) => {
        // A footnote's in-page link would replace the hash holding the page state.
        if (!href.startsWith("#")) return;
        event.preventDefault();
        document
          .getElementById(href.slice(1))
          ?.scrollIntoView({ block: "nearest" });
      }}
    />
  ),
  section: ({ node, ...props }) => (
    <section
      {...props}
      className="mt-3 border-0 border-t border-solid border-line pt-1 text-[0.8rem] text-fg-2"
    />
  ),
};

export function Description({ text }: { text: string }) {
  return (
    <div className="mt-4 leading-relaxed">
      <Markdown
        remarkPlugins={[remarkGfm]}
        remarkRehypeOptions={{
          footnoteLabelProperties: { className: ["sr-only"] },
        }}
        components={components}
      >
        {text}
      </Markdown>
    </div>
  );
}
