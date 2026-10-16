import argparse
import json
import os

from md_parser import parse_markdown
from tex_parser import parse_latex
from tree_schema import FakeTokenizer, LlamaTokenizer, build, preorder

PARSERS = {".md": ("markdown", parse_markdown), ".tex": ("latex", parse_latex)}


def parse_file(path, tok, min_tokens):
    ext = os.path.splitext(path)[1]
    if ext not in PARSERS:
        raise ValueError(f"{path}: unsupported file type {ext!r}, expected one of {sorted(PARSERS)}")
    domain, parse = PARSERS[ext]
    src = open(path, encoding="utf-8").read()
    root = parse(src)
    if "".join(n.text for n in preorder(root)) != src:
        raise ValueError(f"{path}: the tree does not rebuild the source")
    doc = build(root, tok, doc_id=os.path.basename(path), domain=domain, min_tokens=min_tokens)
    if "".join(n["text"] for n in doc["nodes"]) != src:
        raise ValueError(f"{path}: merging broke the reading order")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docs", nargs="+")
    ap.add_argument("--tokenizer", choices=["fake", "llama"], default="fake")
    ap.add_argument("--bin", default="llama.cpp/build/bin/llama-tokenize")
    ap.add_argument("--model", default="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--min-tokens", type=int, default=32)
    a = ap.parse_args()
    tok = FakeTokenizer() if a.tokenizer == "fake" else LlamaTokenizer(a.bin, a.model)
    for path in a.docs:
        doc = parse_file(path, tok, a.min_tokens)
        out = os.path.splitext(path)[0] + ".json"
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        print(out, json.dumps(doc["stats"]))


if __name__ == "__main__":
    main()
