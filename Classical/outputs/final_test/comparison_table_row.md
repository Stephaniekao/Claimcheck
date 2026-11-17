| Model | Overall (strict recall / precision / F1 / AUROC) | Storyboard B slice (strict recall) | Spread across 3 seeds (± std) | Calibration (ECE) | Training cost (GPU hours) | Inference latency (ms per claim) | Inspectability |
|---|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.00 / N/A / N/A / 0.58 | 0.00 | AUROC ± 0.000 | 0.003 (RAGBench) / 0.459 (Inserted) | 0 GPU (0.12 CPU hours) | 2.7 | 7 standardized coefficients |

Strict recall, precision and F1: Inserted Error Set v2 test. AUROC: RAGBench test, answer level. Spread: RAGBench test answer AUROC over seeds 42, 43, 44.
