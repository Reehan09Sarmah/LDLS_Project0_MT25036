import argparse
import fnmatch
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SKIP_DIRS = {".git", "llama.cpp", "models", "__pycache__", ".venv", ".pytest_cache"}
KV_PER_TOKEN = 2 * 16 * 8 * 64 * 2


def repo_files():
    out = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), REPO).replace(os.sep, "/")
            if not rel.endswith(".pyc"):
                out.append(rel)
    return sorted(out)


def read(rel, default=None):
    path = os.path.join(REPO, rel)
    return open(path, encoding="utf-8").read() if os.path.exists(path) else default


def table_body(rel, missing):
    text = read(rel)
    if text is None:
        return f"*{missing}*"
    return "\n".join(l for l in text.splitlines() if l.startswith("|")).strip()


def table_rows(rel):
    text = read(rel) or ""
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in text.splitlines() if l.startswith("|")]
    return [r for r in rows[1:] if not r[0].startswith("---")]


def load_json(rel):
    text = read(rel)
    return json.loads(text[text.index("{"):]) if text and "{" in text else None


def short(text, limit=45):
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def seq_words(seqs):
    return f"sequence {seqs[0]}" if len(seqs) == 1 else "sequences " + ", ".join(map(str, seqs[:-1])) + f" and {seqs[-1]}"


def number(value):
    return "0" if value == 0 else f"{value:.2f}"


def write_zip(dest, names):
    top = os.path.basename(dest)
    with zipfile.ZipFile(dest + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for rel in sorted(names):
            info = zipfile.ZipInfo(f"{top}/{rel}", date_time=(2026, 10, 16, 0, 0, 0))
            info.external_attr = (0o755 if rel.endswith(".sh") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, open(os.path.join(dest, rel), "rb").read())
    return dest + ".zip"


def summary():
    tests = table_rows("results/test_results.md")
    times = table_rows("results/timings.md")
    stats = [json.load(open(p))["stats"] for p in sorted(glob.glob(os.path.join(REPO, "data", "markdown", "*.json")))
             + sorted(glob.glob(os.path.join(REPO, "data", "latex", "*.json")))]
    if not tests or not times or not stats:
        return {}
    ratios = [float(r[6].rstrip("x")) for r in times]
    timing_text = read("results/timings.md")
    cpu = timing_text.split("CPU: ", 1)[1].split(";", 1)[0] if "CPU: " in timing_text else "unknown CPU"
    return {
        "N_DOCS": str(len(tests)),
        "PASS1": str(sum(r[1] == "PASS" for r in tests)),
        "PASS2": str(sum(r[3] == "PASS" for r in tests)),
        "EDIT_DIFF_MAX": number(max(float(r[2]) for r in tests)),
        "EVICT_DIFF_MAX": number(max(float(r[4]) for r in tests)),
        "FIXED_DIFF_MAX": number(max(float(r[5]) for r in tests)),
        "DRIFT_DOCS": str(sum(float(r[4]) > 0 for r in tests)),
        "FREED_MEASURED": f"{sum(float(r[6].split('/')[0]) for r in tests):.2f}",
        "FREED_EXPECTED": f"{sum(float(r[6].split('/')[1]) for r in tests):.2f}",
        "SPEEDUP_MIN": f"{min(ratios):.1f}",
        "SPEEDUP_MAX": f"{max(ratios):.1f}",
        "SPEEDUP_MEDIAN": f"{statistics.median(ratios):.1f}",
        "EVICT_MS_MAX": f"{math.ceil(max(float(r[7]) for r in times) * 10) / 10:.1f}",
        "TOKENS_MIN": str(min(s["total_tokens"] for s in stats)),
        "TOKENS_MAX": str(max(s["total_tokens"] for s in stats)),
        "CPU": cpu,
    }


def test0_text():
    d = load_json("results/test0.json")
    if d is None:
        return "*Not yet produced: run Test 0 (section 11).*"
    first = lambda s: short(s.strip().splitlines()[0]) if s.strip() else ""
    a, b = d["first_logit_diff_together_vs_alone"]
    out = ("| | sequence 0 | sequence 1 |\n|---|---|---|\n"
           f"| prompt length (tokens) | {d['prompt_tokens'][0]} | {d['prompt_tokens'][1]} |\n"
           f"| output, both prompts in one batch | {first(d['together'][0])!r} | {first(d['together'][1])!r} |\n"
           f"| output, each prompt alone | {first(d['alone'][0])!r} | {first(d['alone'][1])!r} |\n"
           f"| largest score difference, first token | {number(a)} | {number(b)} |\n\n"
           f"**Result: the outputs {'are identical' if d['match'] else 'are NOT identical'}.** As a control, the text "
           "of sequence 1 was replaced by a different sentence; the scores of sequence 0 changed by "
           f"{number(d['seq0_logit_diff_when_seq1_text_changes'])}. The two sequences are therefore fully separated.")
    e = load_json("results/test0_short_prompts.json")
    if e:
        out += (f"\n\nWith very short prompts ({e['prompt_tokens'][0]} and {e['prompt_tokens'][1]} tokens), the outputs "
                f"{'were identical' if e['match'] else 'were not identical'}: the first-token scores differed by "
                f"{number(e['first_logit_diff_together_vs_alone'][0])} and "
                f"{number(e['first_logit_diff_together_vs_alone'][1])}, while the control still showed "
                f"{number(e['seq0_logit_diff_when_seq1_text_changes'])}. The cause is that llama.cpp's CPU code "
                "processes tokens in groups of four and handles the remaining tokens with different arithmetic. When "
                "the first prompt's length is not a multiple of four, the tokens fall into different groups in the "
                "joint run and in the separate runs, so the numbers are rounded differently. First prompts of 12, 16 "
                "and 20 tokens gave identical results; first prompts of 6, 7, 9, 13, 15 and 19 tokens did not "
                "(`results/test0_short_prompts.json`).")
    return out


def example_text():
    d = load_json("results/example_run.json")
    if d is None:
        return ""
    expect = d["evicted_tokens"] * KV_PER_TOKEN
    same = lambda x: "yes" if x == d["answer_before"] else "NO"
    return ("**Worked example** (`data/examples/library_guide.json`, a short made-up library guide; "
            "`results/example_run.json`). A = \"Weekends\", B = \"Students\" (edited), C = \"Printing\" (removed).\n\n"
            "| step | answer about A | same answer? | largest score difference |\n|---|---|---|---|\n"
            f"| before any change | {d['answer_before'].strip()!r} | | |\n"
            f"| after editing B (Test 1) | {d['answer_after_edit'].strip()!r} | {same(d['answer_after_edit'])} | "
            f"{number(d['max_logit_diff_edit'])} |\n"
            f"| after removing C (Test 2) | {d['answer_after_evict'].strip()!r} | {same(d['answer_after_evict'])} | "
            f"{number(d['max_logit_diff_evict'])} |\n\n"
            f"Removing C deleted {seq_words(d['evict_seq_ids'])} over positions [{d['evict_position_range'][0]}, "
            f"{d['evict_position_range'][1]}), that is {d['evicted_tokens']} tokens. The memory freed was "
            f"{d['kv_bytes_freed']:,} bytes; the formula gives {d['evicted_tokens']} × 32,768 = {expect:,} bytes, and "
            f"the extra {d['kv_bytes_freed'] - expect} bytes are llama.cpp's bookkeeping.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "..", "submissions"))
    ap.add_argument("--member", help="Pinaksh, Reehan or Ishaan (default: all)")
    ap.add_argument("--only-own", action="store_true", help="copy only the member's own files + shared files")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    a = ap.parse_args()

    own = json.load(open(os.path.join(REPO, "tools", "ownership.json")))
    members = {k: v for k, v in own.items() if k != "shared"}
    member_names = [m["member"].lower() for m in members.values()]
    if a.member and a.member.lower() not in member_names:
        sys.exit(f"unknown member {a.member!r}; expected one of {member_names}")
    files = repo_files()
    owner, problems, missing = {}, [], []
    for folder, m in members.items():
        for pattern, _ in m["files"]:
            hits = [f for f in files if fnmatch.fnmatch(f, pattern)]
            if not hits:
                missing.append(f"{m['member']}: {pattern}")
            for f in hits:
                if f in owner and owner[f] != folder:
                    problems.append(f"{f} is claimed by both {owner[f]} and {folder}")
                owner[f] = folder
    shared = [f for f in files if any(fnmatch.fnmatch(f, p) for p in own["shared"])]
    problems += [f"{f} has no owner: add it to tools/ownership.json" for f in files
                 if f not in owner and f not in shared]
    drafts = [m["member"] for m in members.values()
              if any(mark in (read(f"reports/{m['member']}.md") or "") for mark in ("DRAFT", "<fill"))]
    for x in missing:
        print("  not yet produced:", x)
    for x in drafts:
        print(f"  report still a draft: reports/{x}.md (remove the DRAFT line, fill every <fill: …>)")
    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        sys.exit(1)
    print(f"ownership OK: {len(owner)} member files, {len(shared)} shared files")
    if a.check:
        return

    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = ""
    template = read("reports/README_template.md")
    if not commit:
        template = template.replace(" This repository was generated from our team's main repository at commit "
                                    "`{{COMMIT}}`.", "")
    values = summary()
    n_docs = values.get("N_DOCS", "all")
    personal_reports = {f"reports/{m['member']}.md" for m in members.values()}
    for folder, m in members.items():
        if a.member and m["member"].lower() != a.member.lower():
            continue
        name = m["member"]
        mine = sorted(f for f, o in owner.items() if o == folder)
        others = {f: members[o]["member"] for f, o in owner.items() if o != folder}
        copy = [f for f in files if f not in personal_reports and f != "reports/README_template.md"
                and (not a.only_own or f in mine or f in shared)]
        dest = os.path.join(os.path.abspath(a.out), m["repo"])
        os.makedirs(dest, exist_ok=True)
        for f in copy:
            os.makedirs(os.path.dirname(os.path.join(dest, f)), exist_ok=True)
            shutil.copy2(os.path.join(REPO, f), os.path.join(dest, f))
        personal = "\n".join(l for l in (read(f"reports/{name}.md") or "").splitlines() if not l.startswith("<!-- DRAFT"))
        team = "| teammate | assigned role | files of that role |\n|---|---|---|\n" + "\n".join(
            f"| {x['member']} | {x['role']} | " + ", ".join(f"`{p}`" for p, _ in x["files"] if p not in personal_reports)
            + " |" for k, x in members.items() if k != folder)
        rows = [f"| `{f}` | {'my role' if f in mine else (others[f] + chr(39) + 's role') if f in others else 'shared'} |"
                for f in copy]
        filled = template
        for key, text in {
            "NAME": name, "ROLE": m["role"], "BRIEF_ITEMS": m["brief_items"], "COMMIT": commit,
            "PERSONAL": personal.strip(), "TEAM_TABLE": team,
            "STATS": table_body("results/stats.md", "Not yet produced: run parser/make_stats.py."),
            "TEST0": test0_text(),
            "TEST_RESULTS": example_text() + f"**All {n_docs} documents:**\n\n"
                            + table_body("results/test_results.md", "Not yet produced: run tests/run_tests.py."),
            "TIMINGS": table_body("results/timings.md", "Not yet produced: run tests/run_tests.py."),
            "FILE_TABLE": "| file | belongs to |\n|---|---|\n" + "\n".join(rows),
            **values,
        }.items():
            filled = filled.replace("{{" + key + "}}", text)
        left = sorted(set(p.split("}}")[0] for p in filled.split("{{")[1:]))
        if left:
            print(f"  warning: unfilled placeholders in {name}'s README: {left}")
        open(os.path.join(dest, "README.md"), "w", encoding="utf-8").write(filled)
        open(os.path.join(dest, ".gitignore"), "w").write(read(".gitignore") or "")
        wanted = set(copy) | {"README.md", ".gitignore"}
        for root, dirs, fnames in os.walk(dest, topdown=False):
            if os.sep + ".git" in root or root.endswith(".git"):
                continue
            for fname in fnames:
                rel = os.path.relpath(os.path.join(root, fname), dest).replace(os.sep, "/")
                if rel not in wanted:
                    os.remove(os.path.join(root, fname))
            if root != dest and not os.listdir(root):
                os.rmdir(root)
        print(f"{name}: {len(wanted)} files -> {dest}")
        print(f"{name}: zip -> {write_zip(dest, wanted)}")
    print("\nNext: push each folder to its owner's public repo BEFORE the deadline, then submit the repo link "
          "and the zip.")


if __name__ == "__main__":
    main()
