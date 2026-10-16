import json
import re
import subprocess

MAX_SEQ = 256
SPARE_SEQ = 2


class Node:
    def __init__(self, node_type, text, parent=None, plain=None):
        self.node_type = node_type
        self.text = text
        self.plain = text if plain is None else plain
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)


def preorder(node):
    yield node
    for child in node.children:
        yield from preorder(child)


def subtree_text(node):
    return "".join(n.text for n in preorder(node))


class LlamaTokenizer:
    bos = 128000

    def __init__(self, binary, model):
        self.cmd = [binary, "-m", model, "--stdin", "--ids", "--no-bos", "--no-escape", "--no-parse-special",
                    "--log-disable"]
        self.cache = {}

    def __call__(self, text):
        if not text:
            return []
        if text not in self.cache:
            out = subprocess.run(self.cmd, input=text, capture_output=True, text=True, check=True).stdout
            line = [l for l in out.splitlines() if l.startswith("[")][-1]
            self.cache[text] = json.loads(line)
        return self.cache[text]


class FakeTokenizer:
    bos = 0

    def __call__(self, text):
        return [1 + sum(map(ord, w)) % 1000 for w in re.findall(r"\S+|\n", text)]


def merge_small(root, tok, min_tokens):
    def size(node):
        return len(tok(subtree_text(node)))

    def visit(node):
        for child in node.children:
            visit(child)
        for child in node.children:
            if child.children and size(child) < min_tokens:
                child.text = subtree_text(child)
                child.plain = "".join(n.plain for n in preorder(child))
                child.children = []
        i = 0
        while i < len(node.children):
            child = node.children[i]
            if child.children or size(child) >= min_tokens:
                i += 1
                continue
            prev = node.children[i - 1] if i > 0 else None
            nxt = node.children[i + 1] if i + 1 < len(node.children) else None
            if prev is not None and not prev.children:
                prev.text += child.text
                prev.plain += child.plain
            elif nxt is not None and not nxt.children:
                nxt.text = child.text + nxt.text
                nxt.plain = child.plain + nxt.plain
            elif i == 0:
                node.text += child.text
                node.plain += child.plain
            else:
                i += 1
                continue
            del node.children[i]

    visit(root)


def count_leaves(root):
    return sum(1 for n in preorder(root) if not n.children)


def assign(root, tok):
    nodes = list(preorder(root))
    for i, leaf in enumerate(n for n in nodes if not n.children):
        leaf.seqs = [i]
    for node in reversed(nodes):
        if node.children:
            node.seqs = sorted(s for c in node.children for s in c.seqs)
    for node in nodes:
        node.tokens = ([tok.bos] if node is root else []) + tok(node.text)
        start = 0 if node.parent is None else node.parent.start + len(node.parent.tokens)
        node.start = start
        node.positions = list(range(start, start + len(node.tokens)))
    return nodes


def check_llama_rules(nodes):
    n_seq = 1 + max(s for n in nodes for s in n.seqs)
    if n_seq + SPARE_SEQ > MAX_SEQ:
        raise ValueError("too many sequences")
    last = {}
    for node in nodes:
        if node.parent is not None and not set(node.seqs) <= set(node.parent.seqs):
            raise ValueError("child sequences must be inside parent sequences")
        for p in node.positions:
            for s in node.seqs:
                if p != last.get(s, -1) + 1:
                    raise ValueError(f"sequence {s}: positions must be consecutive")
                last[s] = p
    return n_seq


def stats(root, nodes, tok):
    def depth(node):
        return 1 + max((depth(c) for c in node.children), default=0)

    total = sum(len(n.tokens) for n in nodes)
    markup = sum(max(0, len(tok(n.text)) - len(tok(n.plain))) for n in nodes)
    leaves = sorted(len(tok(subtree_text(n))) for n in nodes if not n.children)
    top = root
    while len(top.children) == 1:
        top = top.children[0]
    return {"total_tokens": total, "depth": depth(root), "n_nodes": len(nodes), "n_seq": len(leaves),
            "leaf_tokens_min": leaves[0], "leaf_tokens_median": leaves[len(leaves) // 2],
            "leaf_tokens_max": leaves[-1], "top_section_tokens": [len(tok(subtree_text(c))) for c in top.children],
            "markup_share": round(markup / max(total, 1), 3)}


def build(root, tok, doc_id, domain, min_tokens=32):
    merge_small(root, tok, min_tokens)
    while count_leaves(root) + SPARE_SEQ > MAX_SEQ:
        min_tokens *= 2
        merge_small(root, tok, min_tokens)
    nodes = assign(root, tok)
    n_seq = check_llama_rules(nodes)
    ids = {id(n): i for i, n in enumerate(nodes)}
    return {"schema_version": 1, "doc_id": doc_id, "domain": domain, "n_seq": n_seq,
            "stats": stats(root, nodes, tok),
            "nodes": [{"id": ids[id(n)], "parent": -1 if n.parent is None else ids[id(n.parent)],
                       "node_type": n.node_type, "text": n.text, "tokens": n.tokens, "positions": n.positions,
                       "seq_ids": n.seqs} for n in nodes]}
