import argparse
import json
import os
import platform
import statistics
import subprocess
import sys

KV_PER_TOKEN = 2 * 16 * 8 * 64 * 2
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def run_case(args, case, extra=()):
    cmd = [args.driver, "-m", args.model, "-t", str(args.threads), "-n", str(args.n_predict), "-d", case["doc"],
           "--ask", str(case["ask_node"]), case["question"], "--edit", str(case["edit_node"]), case["edit_text"],
           "--evict", str(case["evict_node"])] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or "{" not in p.stdout:
        sys.exit(f"driver failed on {case['doc']} (exit {p.returncode}):\n" + "\n".join(p.stderr.splitlines()[-15:]))
    return json.loads(p.stdout[p.stdout.index("{"):])


def check(runs):
    expect = runs[0]["evicted_tokens"] * KV_PER_TOKEN
    test1 = all(r["answer_before"] == r["answer_after_edit"] for r in runs)
    test2 = all(r["answer_before"] == r["answer_after_evict"] and abs(r["kv_bytes_freed"] - expect) <= 0.05 * expect
                for r in runs)
    return test1, test2, expect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default="llama.cpp/build/bin/tree-driver")
    ap.add_argument("--model", default="models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--cases", default="tests/cases.json")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    results, timings, all_ok = [], [], True
    for case in json.load(open(args.cases)):
        runs = [run_case(args, case) for _ in range(args.runs)]
        fixed = run_case(args, case, ["--keep-layout"])
        r = runs[0]
        test1, test2, expect = check(runs + [fixed])
        same = all(x["answer_before"] == r["answer_before"] for x in runs + [fixed])
        all_ok &= test1 and test2 and same
        d1 = max(x["max_logit_diff_edit"] for x in runs)
        d2 = max(x["max_logit_diff_evict"] for x in runs)
        doc = json.load(open(case["doc"]))
        name = f'{doc["domain"]}/{os.path.basename(case["doc"])}'
        rng = r["evict_position_range"]
        results.append(f'| {name} | {"PASS" if test1 else "FAIL"} | {d1:.1e} | {"PASS" if test2 else "FAIL"} | {d2:.1e} | '
                       f'{fixed["max_logit_diff_evict"]:.1e} | {r["kv_bytes_freed"] / 2**20:.2f} / {expect / 2**20:.2f} | '
                       f'seqs {r["evict_seq_ids"]}, [{rng[0]}, {rng[1]}) | {r["answer_before"].strip()[:60]!r} |')
        med = {k: statistics.median(x[k] for x in runs) for k in ["prefill_ms", "ask_ms", "edit_ms", "evict_ms"]}
        tokens = doc["stats"]["total_tokens"]
        timings.append(f'| {name} | {tokens} | {med["prefill_ms"]:.0f} | {tokens / med["prefill_ms"] * 1000:.0f} | '
                       f'{med["ask_ms"]:.0f} | {med["edit_ms"]:.0f} | {med["prefill_ms"] / med["edit_ms"]:.1f}x | '
                       f'{med["evict_ms"]:.3f} |')
        print(results[-1], flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "test_results.md"), "w") as f:
        f.write("# Test results\n\n"
                "**Pass rule (the brief):** the greedy answer about A is *identical* after editing B (Test 1) and after "
                "evicting C (Test 2), in every run; Test 2 also needs the KV memory to drop by evicted tokens x 32 KiB "
                "(within 5%). Eviction = `llama_memory_seq_rm(seq, p0, p1)` for each of the subtree's own sequence IDs "
                f"over [p0, p1). Each case ran {args.runs} times plus once with `--keep-layout`.\n\n"
                "**Logits (diagnostic):** the largest difference of the first answer token's logits. After an edit it is "
                "0. After an eviction it is non-zero only when C lies before A in the cache: llama.cpp then reuses C's "
                "freed cells for the question tokens, and the changed cell order changes the CPU attention arithmetic. "
                "With `--keep-layout` (freed cells held by an invisible placeholder sequence while the question is "
                "asked) the difference is exactly 0, so the drift is numerical, not information from C.\n\n"
                "| document | Test 1 | max logit diff | Test 2 | max logit diff | same, layout fixed | "
                "KV MiB freed / expected | evicted (seq IDs, positions) | answer about A |\n"
                "|---|---|---|---|---|---|---|---|---|\n" + "\n".join(results) + "\n")
    with open(os.path.join(args.out, "timings.md"), "w") as f:
        f.write(f"# CPU timings (median of {args.runs} runs)\n\nCPU: {cpu_name()}, {args.threads} threads; "
                "llama.cpp 10a60cf, CPU-only build; Llama-3.2-1B-Instruct Q4_K_M.\n\n"
                "| document | tokens | prefill ms | prefill tokens/s | ask ms (question + answer) | edit B ms | "
                "full prefill / edit B | evict C ms |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(timings) + "\n")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
