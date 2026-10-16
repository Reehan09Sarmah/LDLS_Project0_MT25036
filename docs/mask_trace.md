# How llama.cpp builds the attention mask from sequence IDs

Owner: Pinaksh · llama.cpp commit `10a60cf` · brief item: "Trace mask construction from sequence IDs"

## Where

`src/llama-kv-cache.cpp`, function `set_input_kq_mask` (called for every micro-batch inside `llama_decode`).

## The rule (simplified from the source)

```cpp
const llama_seq_id seq_id = ubatch->seq_id[i][0];   // query token i looks with its FIRST sequence ID
for (each cache cell j) {
    if (cells.is_empty(j))         -> -INFINITY;    // empty slot
    if (!cells.seq_has(j, seq_id)) -> -INFINITY;    // cell is not in my sequence
    if (cells.pos_get(j) > p1)     -> -INFINITY;    // cell is in my future (causal); p1 = my position
    otherwise                      -> 0.0f;         // visible
}
```

Token *i* may attend to cache cell *j* only if (1) the cell is in use, (2) the cell carries *i*'s **first** sequence ID, and (3) the cell's position ≤ *i*'s position. A `-INFINITY` score gets softmax weight exactly 0.

## What this means for treekv

- A token that belongs to several sequences (root, headings) is **seen by** all of them but **looks through** only its first ID. In a tree this is harmless: a node's ancestors carry a superset of its IDs.
- Sibling subtrees carry disjoint IDs, so they never see each other. That is the tree mask.
- A question must live in **one** sequence. The driver copies the target leaf's sequence into a scratch sequence (`llama_memory_seq_cp`), asks there, then deletes the scratch sequence.

## Seeing it on our document

```bash
LLAMA_KV_CACHE_DEBUG=3 ./llama.cpp/build/bin/tree-driver -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -n 2 -d data/examples/library_guide.json --ask 4 "When does the library close on Sunday?" --evict 7 2> results/kv_debug.log
```

The log prints a map of the cache cells: `.` = empty, a digit = a cell with one sequence, `M` = a cell shared by several sequences (root and headings). `LLAMA_BATCH_DEBUG=1` additionally prints each batch row (token, position, sequence IDs).

## Our cell map (real run, `data/examples/library_guide.json`, 205 tokens)

From `results/kv_debug.log` (`LLAMA_KV_CACHE_DEBUG=3`), written by `find_slot` just before the question is decoded, as run lengths over cache cells 0, 1, 2, …:

```text
used = 205 :  M x54 | 0 x34 | M x34 | 2 x52 | 3 x31
positions  :  0–53  | 54–87 | 54–87 | 36–87 | 36–66
```

| cells | node(s) | sequence IDs on the cells | why |
|---|---|---|---|
| `M` ×54 | root + title + "Opening Hours" | several (0–3, then 0, 1) | ancestors carry the IDs of all leaves below them |
| `0` ×34 | "Weekdays" (leaf) | {0} | one sequence per leaf |
| `M` ×34 | "Weekends" (leaf A) | {1, 4} | its own ID 1 + the scratch ID 4 added by `llama_memory_seq_cp` for the question |
| `2` ×52 | "Borrowing Rules" + "Students" | {2} | Borrowing has one leaf below it |
| `3` ×31 | "Printing" (+ merged "Contact") | {3} | leaf C |

The position row shows the key design rule: **siblings reuse the same position numbers**. Weekdays and Weekends both cover 54–87, and Borrowing and Printing both start at 36. Every sequence still sees gap-free positions along its own root-to-leaf path.

During the answer, the question and answer tokens appear as `4` ×13 (scratch sequence only). After `evict(C)` the next snapshot reads `used = 174` (205 − 31): the `3` run is gone, and nothing else moved.

Applying the rule of this note: a question token in sequence 4 at position ≥ 88 sees exactly the cells carrying 4, which are root, title, Opening Hours and Weekends. Weekdays (`0`), Borrowing/Students (`2`) and Printing (`3`) are masked with −∞.
