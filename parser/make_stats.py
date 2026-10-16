import argparse
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "stats.md"))
    args = ap.parse_args()
    rows = []
    for path in args.docs:
        doc = json.load(open(path))
        s = doc["stats"]
        sections = ", ".join(map(str, s["top_section_tokens"]))
        rows.append(f'| {doc["domain"]}/{os.path.basename(path)} | {s["total_tokens"]} | {s["depth"]} | {s["n_seq"]} | '
                    f'{s["leaf_tokens_min"]} / {s["leaf_tokens_median"]} / {s["leaf_tokens_max"]} | {sections} | '
                    f'{100 * s["markup_share"]:.1f}% |')
    with open(args.out, "w") as f:
        f.write("# Tree statistics\n\nMarkup share = tokens(text) - tokens(text without markup), summed over nodes "
                "(Markdown: syntax removed by regular expressions; LaTeX: pylatexenc's latex-to-text). It is an "
                "estimate.\n\n| document | tokens | depth | sequences | leaf tokens min / median / max | "
                "top-level section sizes | markup share |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
