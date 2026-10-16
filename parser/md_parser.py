import re

from markdown_it import MarkdownIt

from tree_schema import Node


def strip_markdown(text):
    t = re.sub(r"^```.*$", "", text, flags=re.M)
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", "", t, flags=re.M)
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)
    return re.sub(r"[*_`|>]", "", t)


def parse_markdown(src):
    lines = src.splitlines(keepends=True)
    heads = [(int(t.tag[1]), t.map[0]) for t in MarkdownIt("commonmark").parse(src) if t.type == "heading_open"]
    first = heads[0][1] if heads else len(lines)
    head = "".join(lines[:first])
    root = Node("document", head, plain=strip_markdown(head))
    stack = [(0, root)]
    for k, (level, start) in enumerate(heads):
        stop = heads[k + 1][1] if k + 1 < len(heads) else len(lines)
        text = "".join(lines[start:stop])
        while stack[-1][0] >= level:
            stack.pop()
        node = Node(f"h{level}", text, parent=stack[-1][1], plain=strip_markdown(text))
        stack.append((level, node))
    return root
