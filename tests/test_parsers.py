import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "parser"))

from md_parser import parse_markdown
from parse import parse_file
from tex_parser import parse_latex
from tree_schema import MAX_SEQ, SPARE_SEQ, FakeTokenizer, build, preorder

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tok = FakeTokenizer()


def kinds(root):
    return [n.node_type for n in preorder(root)]


def rebuilds(root, src):
    return "".join(n.text for n in preorder(root)) == src


def test_md_code_fence_is_not_a_heading():
    src = "# A\n\ntext\n\n```\n# not a heading\n```\n\n## B\n\nmore\n"
    root = parse_markdown(src)
    assert kinds(root) == ["document", "h1", "h2"]
    assert rebuilds(root, src)


def test_md_without_headings():
    src = "just a paragraph\nand another line\n"
    root = parse_markdown(src)
    assert kinds(root) == ["document"] and rebuilds(root, src)


def test_md_setext_and_skipped_levels():
    src = "Title\n=====\n\nintro\n\n### deep\n\nx\n\n## mid\n\ny\n"
    root = parse_markdown(src)
    h1 = root.children[0]
    assert h1.node_type == "h1" and [c.node_type for c in h1.children] == ["h3", "h2"]
    assert rebuilds(root, src)


def test_md_empty():
    root = parse_markdown("")
    assert kinds(root) == ["document"] and root.text == ""


def test_tex_commented_and_starred_sections():
    src = ("\\documentclass{article}\n\\begin{document}\n% \\section{hidden}\n\\section{A}\na\n"
           "\\section*{B}\nb\n\\end{document}\n")
    root = parse_latex(src)
    assert kinds(root) == ["document", "section", "section"]
    assert rebuilds(root, src)


def test_tex_proof_attaches_to_statement():
    src = ("\\begin{document}\n\\section{S}\n\\begin{theorem}\nT\n\\end{theorem}\n"
           "\\begin{proof}\nP\n\\end{proof}\n\\end{document}\n")
    root = parse_latex(src)
    thm = root.children[0].children[0]
    assert thm.node_type == "theorem" and [c.node_type for c in thm.children] == ["proof"]
    assert rebuilds(root, src)


def test_tex_proof_after_text_does_not_attach():
    src = ("\\begin{document}\n\\section{S}\n\\begin{lemma}\nL\n\\end{lemma}\nSome text.\n"
           "\\begin{proof}\nP\n\\end{proof}\n\\end{document}\n")
    root = parse_latex(src)
    assert [c.node_type for c in root.children[0].children] == ["lemma", "text", "proof"]
    assert rebuilds(root, src)


def test_tex_custom_theorem_environment():
    src = ("\\newtheorem{thm}{Theorem}\n\\begin{document}\n\\section{S}\n\\begin{thm}\nT\n\\end{thm}\n"
           "\\begin{proof}\nP\n\\end{proof}\n\\end{document}\n")
    root = parse_latex(src)
    thm = root.children[0].children[0]
    assert thm.node_type == "thm" and thm.children[0].node_type == "proof"


def test_tex_nested_environment_is_not_cut():
    src = ("\\begin{document}\n\\section{S}\n\\begin{theorem}\n\\begin{equation}x\\end{equation}\n"
           "\\end{theorem}\n\\end{document}\n")
    root = parse_latex(src)
    assert kinds(root) == ["document", "section", "theorem"] and rebuilds(root, src)


def test_tex_fragment_without_document_environment():
    src = "\\section{A}\nx\n\\subsection{B}\ny\n"
    root = parse_latex(src)
    assert kinds(root) == ["document", "section", "subsection"] and rebuilds(root, src)


def test_tex_text_after_end_document_is_kept():
    src = "\\begin{document}\n\\section{A}\nx\n\\end{document}\ntrailing notes\n"
    root = parse_latex(src)
    assert rebuilds(root, src)


def test_tex_empty_and_unbalanced():
    assert rebuilds(parse_latex(""), "")
    src = "\\begin{document}\n\\section{A}\n\\begin{theorem}\nnever closed\n\\end{document}\n"
    assert rebuilds(parse_latex(src), src)


def test_tex_verbatim_hides_sections():
    src = "\\begin{document}\n\\section{A}\n\\begin{verbatim}\n\\section{fake}\n\\end{verbatim}\n\\end{document}\n"
    root = parse_latex(src)
    assert kinds(root) == ["document", "section"] and rebuilds(root, src)


def test_schema_rules_on_deep_tree():
    src = "".join(f"# S{i}\n\n" + "word " * 40 + "\n\n## T{i}\n\n" + "word " * 40 + "\n\n" for i in range(5))
    doc = build(parse_markdown(src), tok, "x", "markdown")
    first = {}
    for n in doc["nodes"]:
        if n["parent"] >= 0:
            parent = doc["nodes"][n["parent"]]
            assert set(n["seq_ids"]) <= set(parent["seq_ids"])
            assert n["positions"][0] == parent["positions"][-1] + 1
        first.setdefault(n["parent"], set()).add(n["positions"][0] if n["positions"] else None)
    assert all(len(v) == 1 for k, v in first.items() if k >= 0)


def test_schema_too_many_sections_are_merged():
    src = "".join(f"# S{i}\n\n" + "word " * 30 + "\n\n" for i in range(400))
    doc = build(parse_markdown(src), tok, "x", "markdown")
    assert doc["n_seq"] + SPARE_SEQ <= MAX_SEQ
    assert "".join(n["text"] for n in doc["nodes"]) == src


def test_schema_tiny_children_merge_but_keep_order():
    src = "# A\n\nx\n\n## b\n\ny\n\n## c\n\nz\n"
    doc = build(parse_markdown(src), tok, "x", "markdown")
    assert doc["n_seq"] == 1 and "".join(n["text"] for n in doc["nodes"]) == src


@pytest.mark.parametrize("folder", ["markdown", "latex", "examples"])
def test_every_sample_document(folder):
    path = os.path.join(ROOT, "data", folder)
    for name in sorted(os.listdir(path)):
        if name.endswith((".md", ".tex")):
            doc = parse_file(os.path.join(path, name), tok, 32)
            assert doc["n_seq"] >= 1


def test_tex_unclosed_document_keeps_trailing_text():
    src = "\\begin{document}\n\\section{A}\n\\begin{figure}\nf\n\\end{figure}\ntrailing paragraph\n"
    root = parse_latex(src)
    assert kinds(root) == ["document", "section", "figure", "text"] and rebuilds(root, src)


def test_tex_chapters_nest_sections():
    src = "\\begin{document}\n\\chapter{One}\nx\n\\section{A}\ny\n\\chapter{Two}\nz\n\\section{B}\nw\n\\end{document}\n"
    root = parse_latex(src)
    assert [c.node_type for c in root.children] == ["chapter", "chapter"]
    assert [c.node_type for c in root.children[0].children] == ["section"]
    assert rebuilds(root, src)


def test_unknown_file_type_is_rejected(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        parse_file(str(path), tok, 32)


def test_sequence_budget_keeps_two_spare():
    src = "".join(f"# S{i}\n\n" + "word " * 40 + "\n\n" for i in range(300))
    doc = build(parse_markdown(src), tok, "x", "markdown")
    assert doc["n_seq"] <= MAX_SEQ - SPARE_SEQ
