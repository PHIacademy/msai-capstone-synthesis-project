"""Module-level curriculum text for the tutor's feedback (reused from Capstone 6, Section 3.2).

Reads ict_curriculum.md (the official Curriculum and Assessment Guide) and keeps, for each
Compulsory Part module A-E, the Introduction and Learning Objectives block. Grounding stops
at module level because the guide's topic table cannot be mapped one-to-one to the 26 sub-topics.
"""
from __future__ import annotations

import re
from pathlib import Path

_HEAD = re.compile(r"\n([A-E])\.\s+([A-Za-z][A-Za-z ’'\-]+?)\n\nIntroduction\n")


def load_curriculum(path: str | Path | None) -> dict:
    """Return {'A': text, ...}; {} when the file is missing or the five module headings are not found."""
    if path is None or not Path(path).exists():
        return {}
    text = Path(path).read_text(encoding="utf-8")
    heads = list(_HEAD.finditer(text))[:5]           # the five Compulsory Part modules come first
    if len(heads) < 5:
        return {}
    bounds = [h.start() for h in heads] + [len(text)]
    out = {}
    for i, h in enumerate(heads):
        block = text[h.end(): bounds[i + 1]]
        cut = block.find("\n\nDetails\n")
        block = block[:cut] if cut != -1 else block
        out[h.group(1)] = re.sub(r"\n{2,}", "\n\n", block).strip()
    return out
