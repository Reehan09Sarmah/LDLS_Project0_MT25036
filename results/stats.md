# Tree statistics

Markup share = tokens(text) - tokens(text without markup), summed over nodes (Markdown: syntax removed by regular expressions; LaTeX: pylatexenc's latex-to-text). It is an estimate.

| document | tokens | depth | sequences | leaf tokens min / median / max | top-level section sizes | markup share |
|---|---|---|---|---|---|---|
| markdown/embedding.json | 690 | 4 | 7 | 47 / 92 / 162 | 123, 348, 195 | 17.2% |
| markdown/llguidance.json | 922 | 3 | 5 | 94 / 123 / 261 | 94, 143, 123, 98, 261 | 31.1% |
| markdown/mtmd.json | 1545 | 3 | 3 | 143 / 190 / 688 | 190, 143, 688 | 37.8% |
| markdown/rpc.json | 1378 | 3 | 6 | 48 / 287 / 416 | 416, 961 | 9.0% |
| markdown/security.json | 1579 | 4 | 8 | 43 / 152 / 275 | 605, 842 | 20.4% |
| latex/kv_cache.json | 802 | 4 | 6 | 36 / 79 / 115 | 55, 121, 306, 161, 79 | 29.6% |
| latex/online_softmax.json | 716 | 4 | 6 | 19 / 75 / 112 | 177, 236, 177, 57 | 26.3% |
| latex/roofline.json | 487 | 4 | 5 | 45 / 61 / 90 | 125, 143, 97, 63 | 24.8% |
| latex/sample2e.json | 1732 | 2 | 2 | 563 / 916 / 916 | 916, 563 | 31.6% |
| latex/tree_mask.json | 587 | 5 | 5 | 33 / 67 / 107 | 116, 222, 183 | 27.6% |
