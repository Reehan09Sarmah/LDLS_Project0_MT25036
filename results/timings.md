# CPU timings (median of 3 runs)

CPU: Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz, 4 threads; llama.cpp 10a60cf, CPU-only build; Llama-3.2-1B-Instruct Q4_K_M.

| document | tokens | prefill ms | prefill tokens/s | ask ms (question + answer) | edit B ms | full prefill / edit B | evict C ms |
|---|---|---|---|---|---|---|---|
| markdown/embedding.json | 690 | 10275 | 67 | 1830 | 1033 | 9.9x | 0.071 |
| markdown/llguidance.json | 922 | 17417 | 53 | 2500 | 1997 | 8.7x | 0.091 |
| markdown/mtmd.json | 1545 | 29990 | 52 | 839 | 3989 | 7.5x | 0.178 |
| markdown/rpc.json | 1378 | 23976 | 57 | 591 | 3585 | 6.7x | 0.083 |
| markdown/security.json | 1579 | 29326 | 54 | 926 | 2266 | 12.9x | 0.287 |
| latex/kv_cache.json | 802 | 14196 | 56 | 1517 | 1966 | 7.2x | 0.035 |
| latex/online_softmax.json | 716 | 13720 | 52 | 1088 | 2147 | 6.4x | 0.078 |
| latex/roofline.json | 487 | 8523 | 57 | 1182 | 1321 | 6.5x | 0.039 |
| latex/sample2e.json | 1732 | 33062 | 52 | 1421 | 17865 | 1.9x | 0.223 |
| latex/tree_mask.json | 587 | 10285 | 57 | 1106 | 859 | 12.0x | 0.077 |
