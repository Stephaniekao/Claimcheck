"""Split an AI answer into claims, one per sentence (Specification Feature 1).

One sentence = one claim, the same unit as RAGBench's `response_sentences` and
the baseline, so results stay comparable across datasets and methods. Every
claim keeps its character offsets in the answer, so the interface can link a
claim back to the sentence it came from.

Rules, in order:
* each non-empty line is split separately; list markers ("- ", "* ", "1. ", "1) ")
  are dropped from the claim but its offsets still point into the answer;
* a sentence ends at . ! or ? (plus any closing quote or bracket) when whitespace
  and then a capital letter, digit, "$", "(" or a quote follows;
* no split after known abbreviations (Inc., Corp., U.S., approx., month names ...)
  or a single capital initial ("J. Smith"). Decimals ("4.1") never split, since
  no space follows the point.

Deterministic, no model and no API. A wrong split can fail silently, as agreed
with the originator, so check_ragbench.py measures agreement with RAGBench's
own sentence split on validation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SPLITTER_VERSION = "1.0.0"

ABBREVIATIONS = {
    "inc", "corp", "co", "ltd", "llc", "plc", "bros", "intl", "mfg", "dept", "no", "nos", "vs", "approx", "est",
    "mr", "ms", "mrs", "dr", "jr", "sr", "st", "fig", "e.g", "i.e", "u.s", "u.k", "u.s.a", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+")
BOUNDARY = re.compile(r"[.!?][\"')\]]*(?=\s+[A-Z0-9$(\"'\u201c])")


@dataclass(frozen=True)
class Claim:
    index: int
    text: str
    start: int  # character offsets in the original answer
    end: int


def _is_abbreviation(text: str, dot: int) -> bool:
    word = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:dot])
    if not word:
        return False
    w = word.group(1).lower().rstrip(".")
    return w in ABBREVIATIONS or (len(w) == 1 and w.isalpha())


def _sentences(line: str, offset: int) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in BOUNDARY.finditer(line):
        if line[m.start()] == "." and _is_abbreviation(line, m.start()):
            continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(line)))
    out = []
    for a, b in spans:
        seg = line[a:b]
        lead = len(seg) - len(seg.lstrip())
        seg = seg.strip()
        if seg:
            out.append((offset + a + lead, offset + a + lead + len(seg)))
    return out


def split(answer: str) -> list[Claim]:
    spans, pos = [], 0
    for line in answer.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        m = LIST_MARKER.match(body)
        skip = m.end() if m else 0
        spans += _sentences(body[skip:], pos + skip)
        pos += len(line)
    return [Claim(i, answer[a:b], a, b) for i, (a, b) in enumerate(spans)]
