# Claim splitter

Feature 1 of the specification (owner: Chengyuan). Splits an AI answer into claims, **one per sentence**, the same unit as RAGBench's `response_sentences` and every method's labels. Each claim keeps its character offsets in the answer, so the interface can link it back to its sentence.

Rule-based and deterministic: no model, no API, no randomness. Rules are listed at the top of `claimcheck_splitter/split.py`.

```bash
python -m unittest discover ClaimSplitter/tests
python ClaimSplitter/check_ragbench.py          # agreement with RAGBench's own split, validation only
python ClaimSplitter/split_answers.py ANSWERS.csv --id-col ID --text-col TEXT --out CLAIMS.csv
```

`check_ragbench.py` reports how often our split matches RAGBench's sentence split and writes up to 30 differing answers per subset to `outputs/`, so mismatches can be read. A wrong split can fail silently (agreed with the originator), which is why this check exists.

`split_answers.py` is how the FinanceBench answers get split before hand labelling. Record `splitter_version` with the labels: if the splitter changes, the claims change, and every method's FinanceBench results change with them (Field 12, entanglement).
