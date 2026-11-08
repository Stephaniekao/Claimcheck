# ClaimCheck

**Human-in-the-loop, claim-level citation verification for financial documents.**

Team: Wan Chi Kao · Chengyuan Liu · Sharon Jennifer Justin Devaraj — CS6170

AI tools can answer questions about 10-K and 10-Q filings, but their answers may use the wrong number, the wrong reporting period, or the wrong business segment, even when they cite a source. ClaimCheck breaks an answer into claims and checks each claim against the filing passage it cites. Each claim gets:

- a label: **Supported**, **Unsupported** or **Uncertain**
- a confidence level (for methods that produce a score)
- a short explanation, with the evidence highlighted

ClaimCheck only advises; it never blocks a claim. "Supported" means *supported by the cited source*, not *verified as true*, and the analyst makes the final call.

## Methods compared

All four methods label the same claims, with the same passages and the same number normalizer (Experiment Card, Fields 3 and 12).

| Row | Method | Owner | Status |
|---|---|---|---|
| Baseline | Rule-based number normalization + keyword matching, plus a rule-based support score | Sharon | Done; test results, support score and error analysis in [Baseline/](Baseline/) |
| Classical | TF-IDF + logistic regression | Stephanie | Planned |
| Deep | DeBERTa-v3-large (NLI checkpoint), fine-tuned on RAGBench FinQA/TAT-QA | Chengyuan | Planned |
| Foundation | LLM judge (Claude; GPT and Gemini if time allows) | Stephanie | Planned |

## Data

| Dataset | Use | Split |
|---|---|---|
| [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) FinQA + TAT-QA | Develop and evaluate verifiers; sentence-level support labels | Official train / validation / test |
| Inserted Error Set v2 (400 claims, 40 companies, 50% Supported) | Correct SEC XBRL claims, plus copies with deliberately inserted errors: Small Numerical (1–5%), Transposed Digits, Flipped Sign, Temporal, Metric Confusion, Unit, and Storyboard B (segment revenue presented as total revenue) | Split by company: 240 / 80 / 80 |
| [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench) | Final evaluation only, never tuning | 147 test + 3 blind showcase examples |

Details on preparation, leakage screening and reproduction are in [Data/README.md](Data/README.md).

### Inserted Error Set Update

The Inserted Error Set has been updated from v1 to v2. The current dataset contains 400 claims from the same 40 companies, with 200 Supported and 200 Unsupported claims, compared to approximately 80% Unsupported in v1.

The updated generation notebook is `Data/data_download_v2.ipynb`. The original notebook is retained for reference. Company assignments and train, validation, and test splits remain unchanged (240 / 80 / 80).

The baseline has been re-run on v2 (rules 1.3.0, which adds a sign check for the new Flipped Sign errors). Its v1 results are archived in `Baseline/outputs/archive/`.

For details on data preparation, error types, splitting, and reproducibility, see [Data/README.md](Data/README.md).

## Evaluation (Experiment Card, Fields 6–10)

- **Main metric:** strict Unsupported recall, i.e. the share of wrong claims labelled Unsupported, reported with precision and F1. Lenient recall also counts Uncertain as caught.
- **Targets:** strict recall ≥ 0.80 and precision ≥ 0.60 on the Inserted Error Set; strict recall ≥ 0.80 on the Storyboard B slice; AUROC ≥ 0.75 on RAGBench; ECE ≤ 0.10. AUROC and ECE apply only to methods that give a score.
- **Uncertain cap:** Uncertain labels may be at most ⅓ of claims. Thresholds and configurations are chosen on validation, with at most 20 tuning trials, and the test split is used once.

## Repository layout

```
Data/
  data_download.ipynb                     RAGBench + FinanceBench download, Inserted Error Set generation/reproduction
  Data Preparation and Split Notes.md
  Inserted Error Set/                     400 claims, locked company split, SEC source snapshots, SHA256 manifest
  RAGBench/                               created by the download step (git-ignored, ~560 MB)
Baseline/                                 rule-based baseline: code, tests, outputs
```

## Getting started

The environment is pinned in the repo: Python **3.14.0** (`.python-version`) and the exact packages in `requirements.txt`. Results were produced on an Apple M5 laptop, CPU only; no GPU is needed.

```bash
# 0. Environment
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt

# Everything in one command (train + validation; never the test splits):
./run_all.sh --quick     # unit tests + smoke test on the Inserted Error Set, a few seconds
./run_all.sh --fetch     # also downloads RAGBench, then runs RAGBench and the support score (~1 min)
./run_all.sh             # same, once RAGBench is in Data/RAGBench/
```

The individual steps, if you want to run them one by one:

```bash
# Smoke test: Inserted Error Set, data already in the repo, a few seconds on a laptop
python3 Baseline/run_inserted_error_set.py

# 1. Get RAGBench into Data/RAGBench/ (either run the notebook or this script;
#    the script checks SHA256 against the team manifest)
python3 Baseline/fetch_ragbench.py

# 2. Run the baseline on train + validation
python3 Baseline/run_inserted_error_set.py
python3 Baseline/run_ragbench.py --no-search     # saved config; without the flag it re-runs the 16-trial search
python3 Baseline/run_support_score.py

# 3. Error analysis from the saved test outputs, then tests
python3 Baseline/error_analysis.py
python3 -m unittest discover Baseline/tests
```

Don't run `--final-test` yourself: each test split is used once, and `run_ragbench.py --final-test` refuses to run again. Train/validation runs rewrite the train/validation files in `Baseline/outputs/` but keep the logged test results. Every results JSON records the commit, the dataset version and the hardware that produced it.

On macOS with the python.org build of Python, `urllib` downloads in the notebook can fail with `CERTIFICATE_VERIFY_FAILED`. Run *Install Certificates.command* in your Python folder, or use `fetch_ragbench.py`, which downloads with `curl`.

## Comparison table (test split)

| Model | Inserted Error Set strict R / P / F1 | Storyboard B strict R | RAGBench claims strict R / P / F1 (FinQA · TAT-QA) | AUROC | Seeds | ECE | Training | Latency | Inspectability |
|---|---|---|---|---|---|---|---|---|---|
| Rule-based baseline (Inserted v2: rules 1.3.0; RAGBench: rules 1.2.0) | 1.00 / 1.00 / 1.00 | 1.00 | 0.18 / 0.05 / 0.08 · 0.05 / 0.02 / 0.03 | 0.57 · 0.46 (support score 1.0.0, added after the test run) | N/A (deterministic) | N/A (score not calibrated) | none | ~1 ms/claim (CPU) | Each label comes from a named rule and the passage line it used |
| *Label everything Unsupported* | *1.00 / 0.50 / 0.67* | *1.00* | *1.00 / 0.02 / 0.03 · 1.00 / 0.01 / 0.02* | | | | | | |
| TF-IDF + LR | | | | | | | | | |
| DeBERTa-v3-large | | | | | | | | | |
| LLM judge | | | | | | | | | |

**Points for the team:**
- **The Inserted Error Set is still saturated by simple rules.** v2's 50/50 labels fixed the trivial reference (flagging everything now fails the precision target), but number and keyword matching still scores 1.00, because the passages list each fact cleanly. Harder passages, such as real 10-K text with distractor lines, would be needed to separate methods here.
- **v2 claims contain two wording giveaways** ("billion" marks every Unit error; odd decimals mark Small Numerical errors). Trained methods should be checked with a claim-only model; see [Data/README.md](Data/README.md).
- **The baseline's support score is near chance on RAGBench** (AUROC 0.57 FinQA, 0.46 TAT-QA), so the 0.75 target is a real step up from number and keyword matching.
- **The baseline labels 73% of RAGBench claims Uncertain,** over the ⅓ cap, because most RAGBench sentences are calculations or contain no number.

Details, error analysis and limitations: [Baseline/README.md](Baseline/README.md).

## Ethics and scope

- Research prototype only; not investment advice.
- Licenses: FinanceBench CC-BY-NC-4.0 (non-commercial); RAGBench CC-BY-4.0.
- Reference answers and labels are kept out of model inputs.
- User feedback is not used as training data.
- User-study records hold no names and are deleted within 30 days after final grades.
