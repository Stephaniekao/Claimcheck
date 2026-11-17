"""DeBERTa-v3-large verifier: windows, scoring and fine-tuning (Experiment Card Field 3).

Start: an NLI checkpoint (entailment / neutral / contradiction). We keep its
head and read P(entailment) as the support score, so the model starts as a
zero-shot NLI checker. Fine-tuning pushes P(entailment) towards 1 for Supported
claims and towards 0 for Unsupported ones (binary log loss on that probability).
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import log_softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .windows import pack

CHECKPOINT = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MAX_LEN = 512
SPECIAL_TOKENS = 3  # [CLS] claim [SEP] window [SEP]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def free_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_name() -> str | None:
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def versions() -> dict:
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "checkpoint": CHECKPOINT, "max_len": MAX_LEN}


class Verifier:
    def __init__(self, path: str = CHECKPOINT, batch_size: int = 32):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device())
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]
        self.batch_size = batch_size
        self._len_cache: dict[int, list[int]] = {}

    # --- windows --------------------------------------------------------
    def _sentence_lengths(self, pid: int, sentences: list[str]) -> list[int]:
        if pid not in self._len_cache:
            enc = self.tok(sentences, add_special_tokens=False)["input_ids"] if sentences else []
            self._len_cache[pid] = [len(x) for x in enc]
        return self._len_cache[pid]

    def windows(self, claim: str, pid: int, passage: dict) -> list[list[int]]:
        budget = MAX_LEN - SPECIAL_TOKENS - len(self.tok(claim, add_special_tokens=False)["input_ids"])
        return pack(self._sentence_lengths(pid, passage["sentences"]), max(budget, 32))

    # --- scoring ----------------------------------------------------------
    @torch.inference_mode()
    def _entail_prob(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        self.model.eval()
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))  # similar lengths per batch
        out = np.empty(len(pairs), dtype=np.float32)
        use_bf16 = device().type == "cuda"
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            enc = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_second",
                           max_length=MAX_LEN, padding=True, return_tensors="pt").to(device())
            with torch.autocast(device_type=device().type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = self.model(**enc).logits.float()
            out[idx] = torch.softmax(logits, dim=-1)[:, self.entail].cpu().numpy()
        return out

    def score(self, claims: pd.DataFrame, passages: list[dict]) -> tuple[pd.DataFrame, float]:
        """Support score per claim = max P(entailment) over its passage windows.

        Returns one row per claim (same order): `score`, and `evidence_used`, the
        text of the window that gave that score, so every label can be traced to
        the passage lines behind it (Field 3, AC3). Also ms per claim.
        """
        self._len_cache.clear()
        pairs, owner, texts = [], [], []
        for i, r in enumerate(claims.itertuples()):
            p = passages[r.answer]
            for w in self.windows(r.claim, r.answer, p):
                text = " ".join(p["sentences"][j] for j in w)
                pairs.append((r.claim, text))
                owner.append(i)
                texts.append(text)
        t0 = time.perf_counter()
        probs = self._entail_prob(pairs)
        ms = (time.perf_counter() - t0) * 1000 / max(len(claims), 1)
        df = pd.DataFrame({"owner": owner, "p": probs, "text": texts})
        best = df.loc[df.groupby("owner")["p"].idxmax()].set_index("owner").reindex(range(len(claims)))
        return pd.DataFrame({"score": best.p.to_numpy(dtype=float), "evidence_used": best.text.to_numpy()}), ms

    def explain(self, claim: str, passage: dict) -> dict:
        """For the interface: the single passage sentence that best supports the claim.

        Scores the claim against each sentence of its best window. Only called for
        the claims a user is looking at, never during evaluation. The returned
        `sentence` is copied from the passage, so it can be highlighted exactly (AC3).
        """
        ws = self.windows(claim, -1, passage)
        self._len_cache.pop(-1, None)
        texts = [" ".join(passage["sentences"][j] for j in w) for w in ws]
        best_w = ws[int(np.argmax(self._entail_prob([(claim, t) for t in texts])))]
        if not best_w:
            return {"key": None, "sentence": None, "sentence_score": float("nan")}
        sp = self._entail_prob([(claim, passage["sentences"][j]) for j in best_w])
        j = best_w[int(np.argmax(sp))]
        return {"key": passage["keys"][j], "sentence": passage["sentences"][j], "sentence_score": float(sp.max())}

    # --- training pairs ---------------------------------------------------
    def training_pairs(self, claims: pd.DataFrame, passages: list[dict], rng: random.Random) -> pd.DataFrame:
        """One (claim, window, y) pair per claim.

        Supported: the window holding most of its annotated supporting sentences
        (train split only); window 0 if none are annotated. Unsupported: a random
        window, since no window should support it.
        """
        self._len_cache.clear()
        rows = []
        for r in claims.itertuples():
            p = passages[r.answer]
            ws = self.windows(r.claim, r.answer, p)
            if r.label == "Supported":
                keys = set(r.support_keys)
                hits = [sum(p["keys"][j] in keys for j in w) for w in ws]
                w = ws[int(np.argmax(hits))] if keys else ws[0]
            else:
                w = ws[rng.randrange(len(ws))]
            rows.append({"claim": r.claim, "window": " ".join(p["sentences"][j] for j in w),
                         "y": 1.0 if r.label == "Supported" else 0.0})
        return pd.DataFrame(rows)

    # --- fine-tuning ------------------------------------------------------
    def fit(self, pairs: pd.DataFrame, lr: float, schedule_epochs: int, seed: int, batch_size: int = 4,
            grad_accum: int = 4, warmup: float = 0.1, weight_decay: float = 0.01, on_epoch_end=None,
            stop_after: int | None = None) -> float:
        """Fine-tune in place on a schedule of `schedule_epochs`; stop early after `stop_after` epochs.

        Calls on_epoch_end(epoch) after each epoch. Returns wall-clock hours on the
        GPU, including any scoring done in on_epoch_end.
        """
        set_seed(seed)
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        steps = math.ceil(len(pairs) / (batch_size * grad_accum)) * schedule_epochs
        sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
        use_bf16 = device().type == "cuda"
        non_entail = [i for i in range(self.model.config.num_labels) if i != self.entail]
        gen = torch.Generator().manual_seed(seed)
        t0 = time.perf_counter()
        for epoch in range(1, schedule_epochs + 1):
            self.model.train()
            perm = torch.randperm(len(pairs), generator=gen).tolist()
            opt.zero_grad()
            batches = math.ceil(len(perm) / batch_size)
            epoch_start = time.perf_counter()
            recent_loss = []
            print(f"Training seed={seed} lr={lr:g} epoch={epoch} pairs={len(perm)}", flush=True)
            for b, s in enumerate(range(0, len(perm), batch_size)):
                rows = pairs.iloc[perm[s:s + batch_size]]
                enc = self.tok(list(rows.claim), list(rows.window), truncation="only_second", max_length=MAX_LEN,
                               padding=True, return_tensors="pt").to(device())
                y = torch.tensor(rows.y.to_numpy(), dtype=torch.float32, device=device())
                with torch.autocast(device_type=device().type, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = self.model(**enc).logits.float()
                ls = log_softmax(logits, dim=-1)
                log_p = ls[:, self.entail]
                log_not_p = torch.logsumexp(ls[:, non_entail], dim=-1)
                loss = -(y * log_p + (1 - y) * log_not_p).mean() / grad_accum
                loss.backward()
                recent_loss.append(loss.detach() * grad_accum)
                if (b + 1) % grad_accum == 0 or s + batch_size >= len(perm):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                    opt.step()
                    sched.step()
                    opt.zero_grad()
                if (b + 1) % 100 == 0 or b + 1 == batches:
                    mean_loss = torch.stack(recent_loss).mean().item()
                    elapsed = (time.perf_counter() - epoch_start) / 60
                    print(f"  batch {b + 1}/{batches}  mean batch loss={mean_loss:.4f}  "
                          f"elapsed={elapsed:.1f} min", flush=True)
                    recent_loss.clear()
            if on_epoch_end is not None:
                on_epoch_end(epoch)
            if stop_after is not None and epoch >= stop_after:
                break
        return (time.perf_counter() - t0) / 3600

    def save(self, path) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
