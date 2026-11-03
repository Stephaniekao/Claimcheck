"""Rule-based claim verifier (Experiment Card, Field 3 - Baseline row).

For each amount in the claim:

1. Supported   - a passage line has the same amount (after normalization)
                 next to the same key words (line item, period, scope).
2. Unsupported - a passage line about the same item and period reports a
                 *different* amount, or the amount is not in the passage at all.
3. Uncertain   - the amount appears, but the key words only partly match
                 (e.g. the claim says "entire revenue" and the passage line
                 says "segment"), or nothing in the claim can be checked.

A claim is Unsupported if any amount is Unsupported, Supported if every amount
is Supported, and Uncertain otherwise. No training, no randomness, no score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import keywords as kw
from .normalize import NORMALIZER_VERSION, Quantity, extract_periods, extract_quantities, quantities_match, years_of

RULES_VERSION = "1.2.0"

SUPPORTED, UNSUPPORTED, UNCERTAIN = "Supported", "Unsupported", "Uncertain"


@dataclass
class Config:
    rel_tol: float = 0.0              # extra relative tolerance on top of rounding
    min_word_overlap: float = 0.5     # fallback keyword overlap when no line item is recognised
    context_window: int = 8           # words on each side of a claim amount used as its key words
    derived_policy: str = "uncertain" # amount not in passage in a sentence that computes something:
                                      # "uncertain" (abstain) or "unsupported" (flag it)


@dataclass
class Unit:
    """One line / sentence of the cited passage."""
    text: str
    quantities: list[Quantity]
    items: set[str]
    periods: set[str]
    scope: str | None
    segments: set[str]
    words: set[str]


@dataclass
class Verdict:
    label: str
    rule: str
    explanation: str
    evidence: list[str] = field(default_factory=list)


# Sentences that compute something ("(136/964) * 100 = 14.09%", "grew by 8.9%")
# produce numbers that are not written in the passage. Plain matching cannot
# check them, so they are Uncertain rather than Unsupported.
_ARITHMETIC = re.compile(r"[=×÷]|\d\s*[*/+]\s*[\d($]|\d\s+[-−–]\s+[$(]?\d|\)\s*[*/]")
_DERIVED_WORDS = re.compile(
    r"\b(?:calculat\w*|comput\w*|ratio|average|mean|percent(?:age)? (?:change|increase|decrease|of)|"
    r"growth(?: rate)?|grew|growing|increase[ds]? (?:by|of)|decrease[ds]? (?:by|of)|declined? by|rose by|fell by|"
    r"change[ds]? (?:by|of|in)|difference|differ|sum|combined|together|total return|return on|cagr|"
    r"approximately|roughly|about|nearly|almost|margin|proportion|share of|portion|times|multipl\w*|divid\w*|"
    r"exceed\w*|more than|less than|greater than|higher than|lower than|at least|at most|over|under|above|below)\b",
    re.I)

# FinQA text has mojibake from curly quotes: "company 2019s" = "company's".
_MOJIBAKE = [(re.compile(r"\b2019(?=s\b)"), "'"), (re.compile(r"\b201[cd]\b"), '"')]


def clean_text(text: str) -> str:
    for pat, rep in _MOJIBAKE:
        text = pat.sub(rep, text)
    return text


def is_derived(claim: str) -> bool:
    return bool(_ARITHMETIC.search(claim) or _DERIVED_WORDS.search(claim))


def _same_kind(a: Quantity, b: Quantity) -> bool:
    return (a.unit in ("%", "bps")) == (b.unit in ("%", "bps"))


def _explode_table(text: str) -> list[str] | None:
    """'Title [["", "2019", "2018"], ["Revenue", "$5", "$4"]]' -> one line per cell:
    'Title Revenue (2019): $5', 'Title Revenue (2018): $4'. Every number then sits
    next to its own row label and column header (period)."""
    start = text.find("[[")
    if start < 0 or not text.rstrip().endswith("]]"):
        return None
    try:
        rows = json.loads(text[start:])
    except ValueError:
        return None
    if not rows or not all(isinstance(r, list) for r in rows):
        return None
    title = text[:start].strip()
    header = [str(c) for c in rows[0]]
    out = [title] if title else []
    for row in rows[1:]:
        cells = [str(c) for c in row]
        label = cells[0] if cells else ""
        for j, cell in enumerate(cells[1:], start=1):
            if not cell.strip():
                continue
            col = header[j] if j < len(header) else ""
            out.append(f"{title} {label} ({col}): {cell}".strip())
        if len(cells) == 1 or not any(c.strip() for c in cells[1:]):
            out.append(f"{title} {label}".strip())
    return out


def split_passage(passage: str | list[str]) -> list[str]:
    lines = passage if isinstance(passage, list) else re.split(r"\n+", passage)
    parts = []
    for line in lines:
        line = clean_text(line.strip())
        if not line:
            continue
        table = _explode_table(line)
        if table is not None:
            parts.extend(table)
        elif len(line) > 300:
            parts.extend(s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(a-z])", line) if s.strip())
        else:
            parts.append(line)
    return parts


def analyse_unit(text: str) -> Unit:
    return Unit(
        text=text,
        quantities=extract_quantities(text),
        items=kw.find_items(text),
        periods=extract_periods(text),
        scope=kw.scope_of(text),
        segments=kw.find_segments(text),
        words=kw.content_words(text),
    )


def _local_context(claim: str, q: Quantity, window: int) -> str:
    left = claim[: q.start].split()[-window:]
    right = claim[q.end:].split()[:window]
    return " ".join(left + right)


def _periods_compatible(claim_p: set[str], unit_p: set[str]) -> bool | None:
    """True/False if both sides name a period, None if either is silent."""
    if not claim_p or not unit_p:
        return None
    claim_dates = {p for p in claim_p if len(p) == 10}
    unit_dates = {p for p in unit_p if len(p) == 10}
    if claim_dates and unit_dates:
        return bool(claim_dates & unit_dates)
    return bool(years_of(claim_p) & years_of(unit_p))


def keyword_match(c_items, c_periods, c_scope, c_segments, c_words, u: Unit, cfg: Config) -> str:
    """'full', 'partial' or 'none' agreement between claim key words and a passage line."""
    if _periods_compatible(c_periods, u.periods) is False:
        return "none"
    if c_segments and u.segments and not (c_segments & u.segments):
        return "none"  # a different segment is a different item

    if c_items:
        if not (c_items & u.items):
            return "none"
        item_level = "full"
    else:
        overlap = len(c_words & u.words) / max(len(c_words), 1)
        if overlap >= cfg.min_word_overlap:
            item_level = "full"
        elif overlap > 0:
            item_level = "partial"
        else:
            return "none"

    # Scope: "entire/total" vs "segment" is the Storyboard B conflict.
    if c_scope and u.scope and c_scope != u.scope:
        return "partial"
    if c_scope == "segment" and c_segments and not (c_segments & u.segments):
        return "partial"
    return item_level


def analyse_passage(passage: str | list[str]) -> list[Unit]:
    """Parse a passage once; pass the result to verify(..., units=) for many claims."""
    return [analyse_unit(t) for t in split_passage(passage)]


def verify(claim: str, passage: str | None, cfg: Config = Config(), units: list[Unit] | None = None) -> Verdict:
    if units is None:
        if not passage or not passage.strip():
            return Verdict(UNCERTAIN, "no_passage", "No cited passage, so the claim cannot be checked.")
        units = analyse_passage(passage)
    if not units:
        return Verdict(UNCERTAIN, "no_passage", "No cited passage, so the claim cannot be checked.")

    claim = clean_text(claim)
    amounts = extract_quantities(claim)
    if not amounts:
        return Verdict(UNCERTAIN, "no_amount", "The claim has no number the rules can read, so it cannot be checked by number matching.")
    claim_periods = extract_periods(claim)
    claim_scope = kw.scope_of(claim)
    claim_segments = kw.find_segments(claim)

    per_amount: list[Verdict] = []
    for q in amounts:
        ctx = _local_context(claim, q, cfg.context_window)
        items = kw.find_items(ctx) or kw.find_items(claim)
        words = kw.content_words(ctx)
        matched = {"full": [], "partial": [], "none": []}
        same_item_other_value = []
        for u in units:
            level = keyword_match(items, claim_periods, claim_scope, claim_segments, words, u, cfg)
            has_amount = any(quantities_match(q, pq, cfg.rel_tol) for pq in u.quantities)
            if has_amount:
                matched[level].append(u)
            elif level == "full" and any(_same_kind(q, pq) for pq in u.quantities):
                same_item_other_value.append(u)
        if matched["partial"] or matched["none"]:
            # The amount is in the passage somewhere. Only call it a conflict when a
            # line names the same item AND the same period explicitly; otherwise a
            # table row without a period would contradict every other row.
            same_item_other_value = [u for u in same_item_other_value
                                     if _periods_compatible(claim_periods, u.periods) is True]

        shown = q.describe()
        derived = cfg.derived_policy == "uncertain" and is_derived(claim)
        if derived:
            same_item_other_value = []
        if matched["full"]:
            v = Verdict(SUPPORTED, "amount_and_keywords_match",
                        f"{shown} appears in the passage next to the same line item and period.",
                        [u.text for u in matched["full"][:2]])
        elif same_item_other_value:
            u = same_item_other_value[0]
            elsewhere = matched["partial"] + matched["none"]
            note = f" {shown} does appear, but on a different line: \"{elsewhere[0].text}\"" if elsewhere else ""
            v = Verdict(UNSUPPORTED, "same_item_different_amount",
                        f"The passage reports {', '.join(x.describe() for x in u.quantities)} for this item and period, not {shown}.{note}",
                        [u.text] + [x.text for x in elsewhere[:1]])
        elif matched["partial"]:
            v = Verdict(UNCERTAIN, "amount_matches_keywords_partial",
                        f"{shown} appears, but the surrounding words only partly match the claim (e.g. total vs segment). Check the scope.",
                        [u.text for u in matched["partial"][:2]])
        elif matched["none"]:
            v = Verdict(UNCERTAIN, "amount_matches_other_item",
                        f"{shown} appears, but next to a different line item or period than the claim names.",
                        [u.text for u in matched["none"][:2]])
        elif derived:
            v = Verdict(UNCERTAIN, "derived_value",
                        f"{shown} is not written in the passage; the claim appears to compute it (change, ratio, sum). "
                        "Number matching cannot check calculations.")
        else:
            v = Verdict(UNSUPPORTED, "amount_not_in_passage",
                        f"{shown} does not appear anywhere in the cited passage (after unit normalization).")
        per_amount.append(v)

    labels = {v.label for v in per_amount}
    if UNSUPPORTED in labels:
        return next(v for v in per_amount if v.label == UNSUPPORTED)
    if labels == {SUPPORTED}:
        return Verdict(SUPPORTED, per_amount[0].rule,
                       " ".join(v.explanation for v in per_amount),
                       [e for v in per_amount for e in v.evidence][:3])
    return next(v for v in per_amount if v.label == UNCERTAIN)


def versions() -> dict[str, str]:
    return {"rules": RULES_VERSION, "normalizer": NORMALIZER_VERSION, "keywords": kw.KEYWORDS_VERSION}
