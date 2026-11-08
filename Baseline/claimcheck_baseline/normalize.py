"""Number, unit and period normalizer.

Shared component (Experiment Card, Field 12): every method uses the same
normalizer, so any change here must bump NORMALIZER_VERSION and be recorded
with each result.

Turns "$4.1 billion", "4,100 million", "4.1B" and "USD 4,100,000,000" into one
value in base units, and keeps track of how precisely the number was written so
that a rounded claim ("$4.1 billion") can match an exact filing value
("4,123 million").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NORMALIZER_VERSION = "1.3.0"  # 1.3.0: Quantity.negative_marked + signs_conflict()

SCALE_WORDS = {
    "thousand": 1e3, "thousands": 1e3, "k": 1e3,
    "million": 1e6, "millions": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
    "billion": 1e9, "billions": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "trillions": 1e12, "tn": 1e12, "t": 1e12,
}
# Single-letter scales are only accepted when glued to the number ("4.1B"),
# otherwise "5 m" could be metres and "Item 7 A" would parse as 7,000.
_SHORT_SCALES = {"k", "m", "b", "t", "mn", "mm", "bn", "tn"}

_DATE_RE = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
# "December 31, 2019" / "dec . 31 , 2019" -> keep the year, drop the day; "dec-13" -> drop.
_TEXT_DAY_RE = re.compile(_MONTHS + r"\.?\s*(\d{1,2})(?:st|nd|rd|th)?\b(?=\s*,?\s*(?:(?:19|20)\d{2})?)", re.I)
_MON_YY_RE = re.compile(r"\b" + _MONTHS + r"[-‐ ]\d{2}\b", re.I)
_YEAR_RE = re.compile(r"\b(?:fy\s?|fiscal\s+(?:year\s+)?)?((?:19|20)\d{2})\b", re.I)

_NUM_RE = re.compile(
    r"""
    (?P<open>\()?                                   # accounting negative "(1,234)"
    (?P<cur>US\$|USD|RMB|CNY|EUR|GBP|JPY|INR|\$|€|£|¥|₹)?\s?
    (?P<sign>[-−–])?\s?
    (?P<cur2>US\$|USD|\$|€|£|¥|₹)?\s?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)(?![\d.,]?\d)
    (?P<glue>\s?)
    (?P<scale>thousands?|millions?|billions?|trillions?|mn|mm|bn|tn|[kmbt])?
    (?![a-z0-9])
    \s?(?P<pct>%|percent\b|per\s+cent\b|percentage\s+points?\b|pp\b|bps\b|basis\s+points?\b)?
    (?P<close>\))?
    """,
    re.I | re.X,
)


@dataclass(frozen=True)
class Quantity:
    value: float            # value in base units (USD, or percent points)
    unit: str | None        # "USD" (any currency), "%", "bps" or None (bare number)
    scaled: bool            # True if a scale word was written ("million")
    half_ulp: float         # half of the last written digit, in base units
    raw: str
    start: int
    end: int
    negative_marked: bool = False  # written with a minus sign attached to it ("-7,085", "USD -7,085")
    bracketed: bool = False        # accounting brackets "(1,234)": may or may not mean negative

    def describe(self) -> str:
        return self.raw.strip()


def strip_dates(text: str) -> str:
    """Blank out dates (ISO, "December 31", "dec-13") so their digits are not read as amounts.

    Four-digit years are kept; extract_quantities() skips them and
    extract_periods() reads them."""
    blank = lambda m: " " * len(m.group(0))
    text = _DATE_RE.sub(blank, text)
    text = _MON_YY_RE.sub(blank, text)
    return _TEXT_DAY_RE.sub(blank, text)


def extract_periods(text: str) -> set[str]:
    """ISO dates and four-digit years mentioned in the text.

    Dates are returned as "YYYY-MM-DD" and also contribute their year, so a
    claim that says "2024" is compatible with a passage that says "2024-12-31".
    """
    periods: set[str] = set()
    for m in _DATE_RE.finditer(text):
        periods.add(m.group(0))
        periods.add(m.group(0)[:4])
    for m in _YEAR_RE.finditer(strip_dates(text)):
        periods.add(m.group(1))
    return periods


def years_of(periods: set[str]) -> set[str]:
    return {p[:4] for p in periods}


def _is_year(m: re.Match, text: str) -> bool:
    num = m.group("num")
    if m.group("cur") or m.group("cur2") or m.group("scale") or m.group("pct"):
        return False
    if "," in num or "." in num or not (1900 <= int(num) <= 2100):
        return False
    return True


def _is_marker(m: re.Match) -> bool:
    """Bare small integers ("( 1 )" footnotes, "2." list items, "3 segments") are not amounts."""
    if m.group("cur") or m.group("cur2") or m.group("scale") or m.group("pct"):
        return False
    num = m.group("num")
    return "," not in num and "." not in num and int(num) < 13


def _minus_is_sign(m: re.Match, text: str) -> bool:
    """True for a minus that makes the number negative ("-7,085", "USD -7,085", "$-5").

    Not counted: subtraction ("$469 - $77", "2019-2018"), list dashes ("- Total ..."),
    en/em dashes used as punctuation, and accounting brackets "(1,234)", which
    tables use for costs and outflows that the text states as positive.
    """
    sign = m.group("sign")
    if sign not in ("-", "−"):
        return False
    i = m.start("sign")
    after = text[m.end("sign"):m.end("sign") + 1]
    if after.isspace():
        return False  # "- 77": a dash or an operator, not a sign
    if m.group("cur"):
        return True  # "$-5", "USD -7,085": the currency belongs to this number
    j = i - 1
    while j >= 0 and text[j] == " ":
        j -= 1
    prev = text[j] if j >= 0 else ""
    return not (prev.isdigit() or prev in ")%$€£¥₹=+-−*/×")


def extract_quantities(text: str) -> list[Quantity]:
    """All monetary / percentage / bare amounts in the text, excluding years and dates."""
    clean = strip_dates(text)
    out: list[Quantity] = []
    for m in _NUM_RE.finditer(clean):
        num = m.group("num")
        if not num or not re.search(r"\d", num):
            continue
        # Skip digits that are part of a word/identifier ("10-K", "Q3", "c-121", "ASC 606").
        before = clean[m.start("num") - 1] if m.start("num") > 0 else " "
        after_idx = m.end("num")
        after = clean[after_idx] if after_idx < len(clean) else " "
        if before.isalpha() or (before == "-" and m.start("num") >= 2 and clean[m.start("num") - 2].isalpha()):
            continue
        if after == "-" and after_idx + 1 < len(clean) and clean[after_idx + 1].isalnum():
            continue  # "10-K", "8-K", accession numbers "0000731766-26-000062"
        if before == "-" and m.start("num") >= 2 and clean[m.start("num") - 2].isdigit():
            continue  # trailing parts of hyphenated identifiers
        if _is_year(m, clean):
            continue
        if after.isalpha() and not m.group("scale") and not m.group("pct"):
            continue  # "2014pullmantur", "3rd"
        if _is_marker(m):
            continue

        scale_word = (m.group("scale") or "").lower()
        if scale_word in _SHORT_SCALES and m.group("glue"):
            scale_word = ""  # "5 m" is not "5 million"
        scale = SCALE_WORDS.get(scale_word, 1.0)

        digits = num.replace(",", "")
        base = float(digits)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        half_ulp = 0.5 * (10 ** -decimals) * scale

        negative = bool(m.group("sign")) or (bool(m.group("open")) and bool(m.group("close")))
        value = -base * scale if negative else base * scale

        pct = (m.group("pct") or "").lower()
        if pct.startswith(("bps", "basis")):
            unit = "bps"
        elif pct:
            unit = "%"
        elif m.group("cur") or m.group("cur2"):
            unit = "USD"
        else:
            unit = None

        end = m.end("scale") if scale_word else m.end("num")
        if pct:
            end = m.end("pct")
        start = m.start("cur") if m.group("cur") else m.start("num")
        marked = _minus_is_sign(m, clean)
        bracketed = bool(m.group("open")) and bool(m.group("close"))
        out.append(Quantity(value, unit, bool(scale_word), half_ulp, text[start:end], start, end, marked, bracketed))
    return out


# Bare numbers in filings usually sit under an "(in millions)" header, so a bare
# passage number may be scaled by any of these factors before comparing.
_IMPLIED_SCALES = (1.0, 1e3, 1e6, 1e9)


def quantities_match(claim: Quantity, passage: Quantity, rel_tol: float = 0.0) -> bool:
    """True if the passage amount supports the claim amount.

    The comparison is rounding-aware: "$4.1 billion" (half_ulp = $0.05bn)
    matches any passage value within $0.05bn. `rel_tol` adds an extra relative
    tolerance on top (tuned on validation; 0 by default).
    Signs are compared by absolute value, because text often carries the sign
    in words ("net cash used in financing activities was $9.3 billion").
    """
    if (claim.unit in ("%", "bps")) != (passage.unit in ("%", "bps")):
        return False
    if claim.unit in ("%", "bps") and claim.unit != passage.unit:
        return False

    c = abs(claim.value)
    candidates = [abs(passage.value)]
    if not passage.scaled and claim.unit not in ("%", "bps"):
        candidates = [abs(passage.value) * s for s in _IMPLIED_SCALES]
    if not claim.scaled and passage.scaled and claim.unit not in ("%", "bps"):
        # Bare claim number ("revenue was 4,907") against "USD 4,907 million".
        candidates += [abs(passage.value) / s for s in _IMPLIED_SCALES[1:]]

    for p in candidates:
        tol = max(claim.half_ulp, rel_tol * max(c, p), 1e-9)
        if abs(c - p) <= tol:
            return True
    return False


# Words that make a plain number negative in meaning ("net cash used in financing
# activities was $11.6 billion"). Bracketed alternatives in a line label, such as
# "Net Income (Loss)" or "Provided by (Used in)", are ignored: they name the line,
# they do not say the value is negative.
_NEGATIVE_WORDS = re.compile(
    r"\b(?:used\s+(?:in|for|by)|net\s+loss(?:es)?|(?:operating|pre-?tax|a|the)\s+loss(?:es)?|loss(?:es)?\s+(?:of|was|were|totaled|amounted)|"
    r"deficits?|outflows?|negative|decrease[ds]?|decreasing|declined?|declines|declining|fell|fall|dropped|reductions?|"
    r"reduced|lower|down|repaid|repayments?|paid|payments?|impairments?|write-?(?:downs?|offs?))\b",
    re.I)

_LABEL_ALTERNATIVE = re.compile(r"\(\s*[A-Za-z][A-Za-z \-]*\)")


def has_negative_wording(text: str) -> bool:
    return bool(_NEGATIVE_WORDS.search(_LABEL_ALTERNATIVE.sub(" ", text)))


def signs_conflict(claim: Quantity, claim_context: str, passage: Quantity, passage_text: str) -> bool:
    """True if one side writes the amount with a minus and the other as a plain positive.

    Example: the filing says "Net Income (Loss) ...: USD 7,085 million" and the
    claim says "USD -7,085 million". The plain side never conflicts when it is in
    accounting brackets "(7,085)" or sits next to negative wording ("used in",
    "net loss", "decreased"), since either can mean the value is negative.
    """
    if claim.negative_marked == passage.negative_marked:
        return False
    plain, plain_text = (passage, passage_text) if claim.negative_marked else (claim, claim_context)
    if plain.bracketed:
        return False
    return not has_negative_wording(plain_text)
