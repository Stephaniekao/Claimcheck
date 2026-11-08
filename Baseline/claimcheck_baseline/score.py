"""Rule-based support score (added after the 1.2.0 test run; see Baseline/README.md).

A number in [0, 1]: how strongly the cited passage supports the claim under the
baseline's own matching rules. It is a *support score*, not a calibrated
probability that the label is right, so ECE does not apply.

The score never disagrees with the label. The label fixes the band, and the
matching features only rank claims inside it:

    Unsupported  [0,   1/3)
    Uncertain    [1/3, 2/3)
    Supported    [2/3, 1]

so thresholds 1/3 and 2/3 give back exactly the rules 1.2.0 labels. A wrong
line item, period or segment can therefore never push a claim into Supported,
however well the number matches.

Inside the band, each claim amount is compared with every passage line on four
features, combined with weights (chosen on RAGBench validation):

    number   the line contains the same amount (after normalization; unit,
             scale and, from rules 1.3.0, sign are part of this check)
    item     same line item (lexicon), or share of the claim's key words on the line
    period   same year / date            (skipped when either side names none)
    scope    same segment / total scope  (skipped when either side names none)

Skipped features drop out and the other weights are rescaled. An amount takes
the score of its best line; a claim takes the score of its weakest amount,
matching the label rule (one Unsupported amount makes the claim Unsupported).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import keywords as kw
from . import rules
from .normalize import extract_quantities

SCORE_VERSION = "1.0.0"

TIER = {rules.UNSUPPORTED: 0, rules.UNCERTAIN: 1, rules.SUPPORTED: 2}
LOWER, UPPER = 1 / 3, 2 / 3  # thresholds that reproduce the labels


@dataclass(frozen=True)
class Weights:
    number: float = 0.50
    item: float = 0.25
    period: float = 0.15
    scope: float = 0.10


# Field 5 budget: 16 RAGBench trials were used for the 1.2.0 config, so 4 remain.
WEIGHT_TRIALS = {
    "proposal": Weights(0.50, 0.25, 0.15, 0.10),
    "item_heavy": Weights(0.40, 0.35, 0.15, 0.10),
    "equal": Weights(0.25, 0.25, 0.25, 0.25),
    "number_heavy": Weights(0.70, 0.10, 0.10, 0.10),
}


def _scope_feature(c: rules.AmountCheck, u: rules.Unit) -> float | None:
    if c.segments and u.segments:
        return 1.0 if c.segments & u.segments else 0.0
    if c.scope and u.scope:
        return 1.0 if c.scope == u.scope else 0.0
    if c.scope == "segment" and c.segments:
        return 0.0  # claim names a segment, the line names none
    return None


def line_score(c: rules.AmountCheck, u: rules.Unit, w: Weights, cfg: rules.Config) -> float:
    """Weighted agreement between one claim amount and one passage line, in [0, 1]."""
    feats = [(w.number, 1.0 if rules.amount_in_unit(c.quantity, c.context, u, cfg) else 0.0)]
    if c.items:
        feats.append((w.item, 1.0 if c.items & u.items else 0.0))
    else:
        feats.append((w.item, len(c.words & u.words) / max(len(c.words), 1)))
    period = rules._periods_compatible(c.periods, u.periods)
    if period is not None:
        feats.append((w.period, 1.0 if period else 0.0))
    scope = _scope_feature(c, u)
    if scope is not None:
        feats.append((w.scope, scope))
    total = sum(wt for wt, _ in feats)
    return sum(wt * v for wt, v in feats) / total if total else 0.0


def _in_band(label: str, f: float) -> float:
    return (TIER[label] + 0.99 * min(max(f, 0.0), 1.0)) / 3


def amount_score(c: rules.AmountCheck, units: list[rules.Unit], w: Weights, cfg: rules.Config) -> float:
    rule = c.verdict.rule
    if rule == "same_item_different_amount":
        f = 0.0  # a line about the same item and period states another amount: the clearest contradiction
    elif rule in ("amount_and_keywords_match", "amount_matches_keywords_partial", "amount_matches_other_item"):
        f = max((line_score(c, u, w, cfg) for u in units
                 if rules.amount_in_unit(c.quantity, c.context, u, cfg)), default=0.0)
    else:  # amount_not_in_passage, derived_value: how close the nearest line comes
        f = max((line_score(c, u, w, cfg) for u in units), default=0.0)
    return _in_band(c.verdict.label, f)


def support_scores(claim: str, passage: str | None, cfg: rules.Config, weights: list[Weights],
                   units: list[rules.Unit] | None = None) -> tuple[list[float], rules.Verdict]:
    """One score per weight setting, plus the verdict (exactly rules.verify())."""
    if units is None:
        units = rules.analyse_passage(passage) if passage and passage.strip() else []
    verdict = rules.verify(claim, passage, cfg, units=units) if units else rules.verify(claim, passage, cfg)
    if not units:
        return [_in_band(verdict.label, 0.0)] * len(weights), verdict
    text = rules.clean_text(claim)
    amounts = extract_quantities(text)
    if not amounts:
        words = kw.content_words(text)
        f = max((len(words & u.words) / max(len(words), 1) for u in units), default=0.0)
        return [_in_band(verdict.label, f)] * len(weights), verdict
    checks = rules.check_amounts(text, amounts, units, cfg)
    return [min(amount_score(c, units, w, cfg) for c in checks) for w in weights], verdict


def support_score(claim: str, passage: str | None, cfg: rules.Config = rules.Config(), w: Weights = Weights(),
                  units: list[rules.Unit] | None = None) -> tuple[float, rules.Verdict]:
    """(score, verdict) for one weight setting."""
    scores, verdict = support_scores(claim, passage, cfg, [w], units)
    return scores[0], verdict


def label_of(score: float) -> str:
    return rules.UNSUPPORTED if score < LOWER else rules.SUPPORTED if score >= UPPER else rules.UNCERTAIN
