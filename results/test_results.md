# Test results

**Pass rule (the brief):** the greedy answer about A is *identical* after editing B (Test 1) and after evicting C (Test 2), in every run; Test 2 also needs the KV memory to drop by evicted tokens x 32 KiB (within 5%). Eviction = `llama_memory_seq_rm(seq, p0, p1)` for each of the subtree's own sequence IDs over [p0, p1). Each case ran 3 times plus once with `--keep-layout`.

**Logits (diagnostic):** the largest difference of the first answer token's logits. After an edit it is 0. After an eviction it is non-zero only when C lies before A in the cache: llama.cpp then reuses C's freed cells for the question tokens, and the changed cell order changes the CPU attention arithmetic. With `--keep-layout` (freed cells held by an invisible placeholder sequence while the question is asked) the difference is exactly 0, so the drift is numerical, not information from C.

| document | Test 1 | max logit diff | Test 2 | max logit diff | same, layout fixed | KV MiB freed / expected | evicted (seq IDs, positions) | answer about A |
|---|---|---|---|---|---|---|---|---|
| markdown/embedding.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 6.10 / 6.09 | seqs [5, 6], [24, 127) | 'The value 2 of --embd-normalize means that the embedding vec' |
| markdown/llguidance.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 8.16 / 8.16 | seqs [4], [203, 464) | '50μs of single-core CPU time for the [JSON Schema Bench](htt' |
| markdown/mtmd.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 21.51 / 21.50 | seqs [2], [524, 1212) | '`llava.cpp' |
| markdown/rpc.json | PASS | 0.0e+00 | PASS | 4.1e-01 | 0.0e+00 | 13.00 / 13.00 | seqs [0], [1, 417) | 'Large tensors' |
| markdown/security.json | PASS | 0.0e+00 | PASS | 3.8e-01 | 0.0e+00 | 18.91 / 18.91 | seqs [0, 1, 2], [132, 513) | 'Implement a sandboxed environment for the model execution' |
| latex/kv_cache.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 2.47 / 2.47 | seqs [5], [80, 159) | 'Because each token contributes one key and one value, which ' |
| latex/online_softmax.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 5.53 / 5.53 | seqs [3, 4], [68, 191) | '$e^{m_A - m}$' |
| latex/roofline.json | PASS | 0.0e+00 | PASS | 0.0e+00 | 0.0e+00 | 3.03 / 3.03 | seqs [3], [58, 155) | '$\\frac{Q}{B} \\cdot I$ seconds' |
| latex/sample2e.json | PASS | 0.0e+00 | PASS | 4.4e-01 | 0.0e+00 | 29.10 / 29.09 | seqs [0], [253, 1184) | 'Displayed text is set apart from ordinary text by being' |
| latex/tree_mask.json | PASS | 0.0e+00 | PASS | 3.6e-01 | 0.0e+00 | 7.41 / 7.41 | seqs [1, 2], [66, 255) | 'Because it is not on the path of $a$' |
