import glob
import html
import json
import os
import re
import subprocess

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def table_rows(path):
    rows = []
    for line in open(os.path.join(ROOT, path)):
        if line.startswith("|") and not line.startswith("|---"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows[1:]


def cell(text):
    return f"<td>{html.escape(text)}</td>"


def main():
    docs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "markdown", "*.json"))) + \
            sorted(glob.glob(os.path.join(ROOT, "data", "latex", "*.json"))):
        d = json.load(open(path))
        docs.append((f'{"md" if d["domain"] == "markdown" else "la"}/{os.path.basename(path)[:-5]}', d["stats"]))
    stats = "".join(
        "<tr>" + cell(name) + cell(str(s["total_tokens"])) + cell(str(s["depth"])) + cell(str(s["n_seq"])) +
        cell(f'{s["leaf_tokens_min"]}/{s["leaf_tokens_median"]}/{s["leaf_tokens_max"]}') +
        cell(f'{100 * s["markup_share"]:.0f}%') + "</tr>" for name, s in docs)

    tests = table_rows("results/test_results.md")
    pass1 = sum(r[1] == "PASS" for r in tests)
    pass2 = sum(r[3] == "PASS" for r in tests)
    edit_diff = max(float(r[2]) for r in tests)
    evict_diff = max(float(r[4]) for r in tests)
    fixed_diff = max(float(r[5]) for r in tests)
    freed = [tuple(map(float, r[6].split("/"))) for r in tests]
    test0 = json.load(open(os.path.join(ROOT, "results", "test0.json")))
    short = json.load(open(os.path.join(ROOT, "results", "test0_short_prompts.json")))

    times = table_rows("results/timings.md")
    cpu = re.search(r"CPU: ([^;]+);", open(os.path.join(ROOT, "results", "timings.md")).read()).group(1)
    timing = "".join(
        "<tr>" + cell(r[0].replace("markdown/", "md/").replace("latex/", "la/").replace(".json", "")) + cell(r[1]) +
        cell(r[2]) + cell(r[5]) + cell(r[6]) + cell(r[7]) + "</tr>" for r in times)

    page = f"""<!doctype html><html><head><meta charset="utf-8"><style>
@page {{ size: 1280px 720px; margin: 0 }}
html, body {{ width: 1280px; height: 720px; overflow: hidden; }}
body {{ margin: 0; font-family: 'DejaVu Sans', sans-serif; color: #1d2433; }}
h1 {{ margin: 18px 28px 2px; font-size: 26px; }}
.sub {{ margin: 0 28px 10px; color: #5b6475; font-size: 14px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 0 28px; }}
.box {{ border: 2px solid #c9cdd6; border-radius: 10px; padding: 8px 12px; font-size: 12.5px; height: 278px; overflow: hidden; }}
.box h2 {{ font-size: 15px; margin: 0 0 6px; color: #1f4e99; }}
table {{ border-collapse: collapse; width: 100%; font-size: 11.5px; }}
td, th {{ border-bottom: 1px solid #e3e6ec; padding: 2px 4px; text-align: left; }}
th {{ color: #5b6475; }}
.big {{ font-size: 17px; font-weight: bold; color: #2e7d32; }}
ul {{ margin: 4px 0; padding-left: 18px; }} li {{ margin: 2px 0; }}
</style></head><body>
<h1>treekv: a tree-structured KV cache in llama.cpp</h1>
<div class="sub">Markdown + LaTeX · Llama-3.2-1B-Instruct Q4_K_M · llama.cpp 10a60cf, CPU only · Pinaksh, Reehan, Ishaan</div>
<div class="grid">
<div class="box"><h2>Domain and method</h2><ul>
<li><b>Markdown:</b> 5 llama.cpp docs; units = sections by heading level.</li>
<li><b>LaTeX:</b> TeX Live sample2e + 4 math notes; units = sections, theorem/lemma/definition/proof/figure/table environments (pylatexenc). A proof becomes the child of the statement it proves.</li>
<li>One sequence ID per leaf; every ancestor carries all its leaves' IDs, so llama.cpp's mask = the tree mask.</li>
<li>Children start right after their parent: siblings share positions, every root-to-leaf path is gap-free.</li>
<li>Subtrees under 32 tokens are merged (at most 254 leaves, keeping 2 sequences for the question and the layout check).</li>
<li>Evict = <code>seq_rm(seq, p0, p1)</code> for the subtree's own sequence IDs.</li></ul></div>
<div class="box"><h2>Tree statistics</h2><table><tr><th>doc</th><th>tokens</th><th>depth</th><th>seqs</th><th>leaf min/med/max</th><th>markup</th></tr>{stats}</table></div>
<div class="box"><h2>Test results (3 runs each, all answers identical across runs)</h2>
<div><span class="big">Test 1 (edit B): {pass1}/{len(tests)} PASS</span> &nbsp; <span class="big">Test 2 (evict C): {pass2}/{len(tests)} PASS</span></div>
<ul><li>Answers about A identical after every edit and every eviction, in all runs. First-token logits: edit <b>{edit_diff:.1e}</b>; evict up to {evict_diff:.1e} when C precedes A (llama.cpp reuses C's freed cells for the question, so the CPU arithmetic order changes), <b>{fixed_diff:.1e}</b> with the cache layout held fixed.</li>
<li>KV memory freed: <b>{sum(f[0] for f in freed):.2f} MiB</b> measured vs {sum(f[1] for f in freed):.2f} MiB expected (evicted tokens × 32 KiB; +12 bytes/cell bookkeeping).</li>
<li><b>Test 0:</b> two prompts in one batch = separate runs: <b>{"match" if test0["match"] else "NO match"}</b> ({test0["prompt_tokens"][0]}/{test0["prompt_tokens"][1]} tokens, logit diff {max(test0["first_logit_diff_together_vs_alone"]):.1e}); changing sequence 1 moves sequence 0 by {test0["seq0_logit_diff_when_seq1_text_changes"]:.1e}.</li>
<li>Finding: with {short["prompt_tokens"][0]}-token prompts the runs differ by {max(short["first_logit_diff_together_vs_alone"]):.2f}: CPU kernels tile 4 tokens at a time, so unaligned batches round differently; isolation stays exact.</li></ul></div>
<div class="box"><h2>CPU run times (median of 3) · {html.escape(cpu)}</h2><table><tr><th>doc</th><th>tokens</th><th>prefill ms</th><th>edit B ms</th><th>full / edit</th><th>evict ms</th></tr>{timing}</table></div>
</div></body></html>"""
    out_html = os.path.join(ROOT, "slide", "slide.html")
    open(out_html, "w", encoding="utf-8").write(page)
    try:
        subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                        f"--print-to-pdf={os.path.join(ROOT, 'slide', 'slide.pdf')}", "file://" + out_html],
                       check=True, capture_output=True)
        print("slide/slide.pdf")
    except (OSError, subprocess.CalledProcessError):
        print("slide/slide.html written; open it in a browser and print to PDF (google-chrome not found)")


if __name__ == "__main__":
    main()
