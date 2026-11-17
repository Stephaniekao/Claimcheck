"""Fit a long passage into DeBERTa's 512-token input (Experiment Card Field 3).

RAGBench passages often exceed 512 tokens once the claim is added. Cutting the
passage at 512 tokens can drop the very line that supports the claim, so we
pack whole sentences into windows that fit, score the claim against every
window, and keep the highest support score: a claim is supported if any part
of the cited passage supports it.
"""
from __future__ import annotations


def pack(lengths: list[int], budget: int) -> list[list[int]]:
    """Greedy, in order: indices of the sentences in each window.

    A sentence longer than the budget gets a window of its own (the tokenizer
    truncates it). No sentence is dropped and the order is kept.
    """
    windows, cur, used = [], [], 0
    for i, n in enumerate(lengths):
        if cur and used + n > budget:
            windows.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        windows.append(cur)
    return windows or [[]]
