# treekv document schema (v1)

Owner: Reehan · approved by: Pinaksh (driver) · status: **frozen for week 1**

One JSON file per document, written by `parser/parse.py` (Markdown → `parser/md_parser.py`, LaTeX → `parser/tex_parser.py`, then `parser/tree_schema.py`) and read by the driver (`driver/tree_driver.cpp`).

## Top level

| field | type | meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `doc_id` | string | source file name |
| `domain` | string | `"markdown"` or `"latex"` |
| `n_seq` | int | number of sequences = number of leaves (the driver adds 2 more: a scratch sequence for questions and a placeholder sequence for `--keep-layout`) |
| `stats` | object | optional: `total_tokens`, `depth`, `n_nodes`, `n_seq`, `leaf_tokens_min/median/max`, `top_section_tokens` (size of each top-level section), `markup_share` |
| `nodes` | array | the tree, **in pre-order** (every parent appears before its children) |

## Node fields (the five required by the brief, plus two extras)

| field | type | rule |
|---|---|---|
| `tokens` | int[] | Llama-3.2 token IDs of the node's **own** text, from `llama-tokenize` with the same GGUF file. The root starts with BOS (128000). |
| `positions` | int[] | one per token, consecutive. Start = parent's start + parent's token count; root starts at 0. **Siblings therefore share position numbers.** |
| `seq_ids` | int[] | a leaf has exactly one ID (leaves are numbered 0, 1, 2, … in reading order); an inner node has the IDs of **all** leaves below it |
| `node_type` | string | `document`; Markdown: `h1` … `h6`; LaTeX: `chapter`, `section`, `subsection`, `subsubsection`, an environment name (`abstract`, `theorem`, `lemma`, `proposition`, `corollary`, `definition`, `remark`, `example`, `proof`, `figure`, `table`, `algorithm`, or any `\newtheorem` name), or `text` |
| `parent` | int | parent's `id`, `-1` for the root |
| `id` | int | extra: index in `nodes` |
| `text` | string | extra: the node's own text; concatenating all `text` in order rebuilds the source exactly |

## How each format becomes a tree

- **Markdown:** every heading starts a node holding the heading and the text up to the next heading. The heading level decides the parent. A `#` inside a code block is not a heading (markdown-it-py).
- **LaTeX:** parsed with pylatexenc, using only the top level of the `document` environment.
  - `\chapter`, `\section`, `\subsection` and `\subsubsection` (also starred) start nodes; the level decides the parent.
  - Each unit environment (list above) becomes its own node, a child of the current section.
  - Paragraphs between environments become `text` nodes.
  - A `proof` that directly follows a theorem, lemma, proposition, corollary or custom theorem environment becomes that statement's **child**, so a question about the proof also sees the statement.
  - Commented-out commands are ignored, and environments nested inside other environments are not cut.
  - Whitespace-only gaps and the closing `\end{document}` stay with the preceding node.

## Merge rules for small subtrees

Applied before tokens, IDs and positions are assigned. `min_tokens = 32` (real tokenizer).

1. **Collapse:** a subtree with fewer than `min_tokens` tokens becomes one leaf.
2. **Join a neighbour:** a small leaf joins the adjacent leaf sibling (the previous one first, otherwise the next one). If it is the first child and has no leaf neighbour, it joins its parent's own text. Otherwise it stays a small leaf (in the ten documents this happens once: a 19-token closing paragraph in `online_softmax.tex`).
3. **Budget:** while leaves + 2 > 256 (llama.cpp's `LLAMA_MAX_SEQ`; the driver needs one scratch sequence for questions and one placeholder sequence for `--keep-layout`), double `min_tokens` and merge again.

All three rules keep reading order intact.

## Validity checks (run by `check_llama_rules`)

- every node's `seq_ids` ⊆ its parent's `seq_ids`;
- along every root-to-leaf path, positions run 0, 1, 2, … with no gap;
- `n_seq + 2 ≤ 256`;
- each document has `stats.total_tokens < 2000`;
- `parse.py` checks that the node texts, joined in order, rebuild the source file exactly (before and after merging).

## Why siblings share position numbers

llama.cpp (commit `10a60cf`, `src/llama-batch.cpp`) requires each sequence's positions to be consecutive. Numbering the whole document in reading order makes a later sibling's sequence jump, and llama.cpp rejects the batch with `sequence N positions are not continuous`. RoPE scores depend only on the distance between tokens, and siblings never see each other, so each branch reads like an ordinary document.

Consequence for the driver: a subtree is evicted by **its own sequence IDs + its position range**, `llama_memory_seq_rm(s, p0, p1)`, never with `s = -1`.

## Example (shortened)

```json
{"schema_version": 1, "doc_id": "library_guide.md", "domain": "markdown", "n_seq": 4,
 "nodes": [
  {"id": 0, "parent": -1, "node_type": "document", "text": "", "tokens": [128000], "positions": [0], "seq_ids": [0, 1, 2, 3]},
  {"id": 4, "parent": 2, "node_type": "h3", "text": "### Weekends\n\n…", "tokens": [ … ], "positions": [55, 56, …], "seq_ids": [1]}
 ]}
```
