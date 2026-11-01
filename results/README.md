# Results

This directory contains public-safe prediction records and summaries for 84
validated retrieval and chain-of-thought experiments.

## Contents

- `predictions/`: one JSONL file per experiment, with 500 ordered test records.
- `summary/all_results.csv`: experiment-level accuracy, micro-F1, macro-F1, and
  weighted-F1.
- `summary/per_relation_results.csv`: precision, recall, F1, and support for
  each of the eight relations.
- `summary/confusion_matrices.json`: one labelled confusion matrix per
  experiment.

Every prediction record contains only an opaque example ID, gold and predicted
labels, and experiment metadata.

Regenerate all summaries with:

```bash
clinical-re summarize-results \
  --predictions-dir results/predictions \
  --output-dir results/summary
```

The command rejects files that do not contain exactly 500 ordered test IDs or
that disagree with the language-specific gold-label sequence.
