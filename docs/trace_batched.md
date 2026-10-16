# Trace of llama.cpp's `batched` example

Owner: Ishaan · llama.cpp commit `10a60cf` · brief item: "Read llama.h and trace the batched example"

File: `examples/batched/batched.cpp`. Run:

```bash
LLAMA_BATCH_DEBUG=1 ./llama.cpp/build/bin/llama-batched -v -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "Hello my name is" -np 2 --kv-unified 2> results/batched_debug.log
```

`-v` is required: `llama-batched` uses llama.cpp's common logger, which hides debug-level messages (including the `LLAMA_BATCH_DEBUG` batch dump) unless the verbosity is raised. `tree-driver` uses the plain `llama.h` logger, which prints them without `-v`.

## What it does, step by step

| # | Step in `batched.cpp` | `llama.h` call(s) |
|---|---|---|
| 1 | load the model | `llama_backend_init`, `llama_model_load_from_file`, `llama_model_get_vocab` |
| 2 | tokenize the prompt | `common_tokenize` → `llama_tokenize` |
| 3 | create a context big enough for the prompt + all continuations (one shared KV cache) | `llama_init_from_model` (`n_ctx`, `n_batch`) |
| 4 | add **every prompt token once, with all sequence IDs** `{0, …, n_parallel−1}`; logits only on the last one | `batch.add(token, pos, seq_ids, false)` → fills `llama_batch` |
| 5 | run the prompt | `llama_decode` |
| 6 | loop: for each sequence *i*, sample its next token from its own logits (`i_batch[i]`); add it with **only** `seq_id = i` at position `n_cur`; decode all sequences together | `llama_sampler_sample(sampler, ctx, i_batch[i])`, `batch.add(new_token_id, n_cur, i, true)`, `llama_decode` |
| 7 | print each sequence and the timings | `common_token_to_piece` (wraps `llama_token_to_piece`), `llama_perf_sampler_print`, `llama_perf_context_print` |

## Why it matters for treekv

`batched` is a tree of depth 1: the prompt is the root (carrying every sequence ID) and each continuation is a leaf (one ID). treekv makes the tree deeper: ancestors carry many IDs, leaves carry one. The `--kv-unified` flag is required because one token belongs to several sequences; without it llama.cpp stops with `sequential split is not supported when there are coupled sequences … (you may need to use the -kvu flag)`.

## What the debug log shows

From `results/batched_debug.log` (real run, Llama-3.2-1B-Instruct Q4_K_M, `-np 2`):

```text
--- batch 1 (the prompt, decoded once) ---
0: id = 128000 (<|begin_of_text|>), pos = 0, n_seq_id = 2, seq_id = [01], output = 0
1: id =   9906 (Hello),             pos = 1, n_seq_id = 2, seq_id = [01], output = 0
2: id =    856 (my),                pos = 2, n_seq_id = 2, seq_id = [01], output = 0
3: id =    836 (name),              pos = 3, n_seq_id = 2, seq_id = [01], output = 0
4: id =    374 (is),                pos = 4, n_seq_id = 2, seq_id = [01], output = 1
--- batch 2 (first generated token of each stream) ---
0: id =  79663 (Sofia),             pos = 5, n_seq_id = 1, seq_id = [0.], output = 1
1: id =  14596 (Bob),               pos = 5, n_seq_id = 1, seq_id = [.1], output = 1
```

- Every prompt token belongs to **both** sequences (`n_seq_id = 2`), so the prompt is computed once and shared, like the root of our tree.
- The two continuations each carry **one** sequence and **share position 5**, like two sibling sections in treekv.
- Only the last prompt token and the generated tokens request logits (`output = 1`).

The final texts were "Hello my name is Sofia and I am so glad you …" (sequence 0) and "Hello my name is Bob and I love to cook." (sequence 1).
