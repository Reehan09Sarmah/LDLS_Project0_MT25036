import re

from pylatexenc.latex2text import LatexNodes2Text
from pylatexenc.latexwalker import LatexEnvironmentNode, LatexMacroNode, LatexWalker

from tree_schema import Node

LEVELS = {"chapter": 1, "section": 2, "subsection": 3, "subsubsection": 4}
UNITS = {"abstract", "theorem", "lemma", "proposition", "corollary", "definition", "remark", "example",
         "proof", "figure", "table", "algorithm"}
STATEMENTS = {"theorem", "lemma", "proposition", "corollary"}


def strip_latex(text):
    try:
        return LatexNodes2Text().latex_to_text(text)
    except Exception:
        return text


def find_cuts(src):
    custom = set(re.findall(r"\\newtheorem\*?\{(\w+)\}", src))
    nodes, _, _ = LatexWalker(src, tolerant_parsing=True).get_latex_nodes()
    doc = next((n for n in nodes if isinstance(n, LatexEnvironmentNode) and n.environmentname == "document"), None)
    body = doc.nodelist if doc else nodes
    limit = src.rfind("\\end{document}", 0, doc.pos + doc.len) if doc else -1
    limit = limit if limit >= 0 else len(src)
    cuts = {}
    for n in body:
        if isinstance(n, LatexMacroNode) and n.macroname in LEVELS:
            cuts[n.pos] = ("section", n.macroname)
        elif isinstance(n, LatexEnvironmentNode) and n.environmentname in UNITS | custom:
            cuts[n.pos] = ("env", n.environmentname)
            cuts.setdefault(n.pos + n.len, ("text", "text"))
    cuts = sorted(cuts.items())
    stops = [pos for pos, _ in cuts[1:]] + [limit]
    keep = [c for c, stop in zip(cuts, stops) if c[1][0] != "text" or src[c[0]:max(stop, c[0])].strip()]
    return keep, custom


def parse_latex(src):
    cuts, custom = find_cuts(src)
    starts = [pos for pos, _ in cuts]
    stops = starts[1:] + [len(src)]
    head = src[:starts[0]] if cuts else src
    root = Node("document", head, plain=strip_latex(head))
    stack = [(0, root)]
    last = root
    for (start, (kind, name)), stop in zip(cuts, stops):
        text = src[start:stop]
        if not text.strip():
            last.text += text
            last.plain += text
            continue
        if kind == "section":
            while stack[-1][0] >= LEVELS[name]:
                stack.pop()
            node = Node(name, text, stack[-1][1], plain=strip_latex(text))
            stack.append((LEVELS[name], node))
        elif name == "proof" and last.node_type in STATEMENTS | custom and not last.children:
            node = Node(name, text, last, plain=strip_latex(text))
        else:
            node = Node(name, text, stack[-1][1], plain=strip_latex(text))
        last = node
    return root
