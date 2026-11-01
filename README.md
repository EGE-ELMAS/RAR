# Bilingual Clinical Relation Extraction

Research code for English–Turkish clinical relation extraction with supervised
baselines and retrieval-augmented prompting. The repository implements random,
KATE, Fine-Tuned Relation Representation (FT-RR/PURE), and Relation-Aware
Retrieval (RAR), together with BERT and PURE classifiers and multiple
chain-of-thought prompt strategies.

## Task

Given a sentence and two gold entity mentions, predict one of eight relation
types from the 2010 i2b2/VA task:

| Label | Relation |
| --- | --- |
| `TrIP` | Treatment improves a medical problem |
| `TrWP` | Treatment worsens a medical problem |
| `TrCP` | Treatment causes a medical problem |
| `TrAP` | Treatment is administered for a medical problem |
| `TrNAP` | Treatment is not administered because of a medical problem |
| `PIP` | One medical problem indicates another |
| `TeRP` | Test reveals a medical problem |
| `TeCP` | Test is conducted to investigate a medical problem |

The experiments use aligned English and Turkish subsets with 1,500 training
and 500 official test examples per language. A stratified development partition
is drawn only from the training pool; the official test set is never used for
checkpoint selection.

## Methods

- **Random:** deterministic query-specific sampling.
- **KATE:** cosine similarity between first-token sentence representations.
- **FT-RR/PURE:** cosine similarity between concatenated subject/object marker
  representations.
- **RAR:** task-specific supervised SimCSE training with sentence, entity, and
  relation-label similarity. Five positive and five negative examples are used
  per anchor. Retrieval combines sentence and entity similarity and produces an ordered demonstration ranking.
- **Supervised baselines:** BERT sequence classification and PURE-style
  relation classification.
- **Prompting:** standard few-shot; zero-shot and few-shot static CoT;
  self-questioning; gold-label-induced CoT; and output-format CoT.

For a `k`-shot experiment, the first `k` IDs are always taken from the retained
ranking. Similarity is computed against the complete authorized training pool.

## Repository layout

```text
clinical-re-bilingual/
├── config/                    # Checkpoints and experiment hyperparameters
├── data/                      # Restricted local data; contents are not distributed
├── notebooks/                 # Concise entry points to tested workflows
├── results/                   # Public-safe predictions and aggregate metrics
├── src/clinical_re_bilingual/
│   ├── data/                  # i2b2 parsing and split manifests
│   ├── evaluation/            # Metrics and confusion matrices
│   ├── inference/             # Hosted-model adapters and execution
│   ├── models/                # BERT and PURE baselines
│   ├── prompting/             # Bilingual prompt construction
│   ├── retrieval/             # Random, KATE, FT-RR, and RAR
│   └── workflows/             # End-to-end experiment workflows
└── tests/                     # Synthetic, non-clinical tests
```

## Installation

Python 3.10 or newer is required. Python 3.12 is recommended.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks,openai,gemini]"
python -m pytest
```

Provider credentials are read from environment variables. Copy `.env.example`
for the variable names, but export values in your shell or secret manager; the
package does not load or commit `.env` files.

## Restricted data

The 2010 i2b2/VA corpus is distributed under a data use agreement and is not
included here. Obtain authorized access from the dataset provider. This
repository contains code, opaque identifiers, labels, and aggregate metrics;
clinical text and text-bearing derivatives remain outside version control.

Expected local files after preprocessing:

```text
data/processed/en_training_pool.tsv
data/processed/en_official_test.tsv
data/processed/tr_training_pool.tsv
data/processed/tr_official_test.tsv
data/manifests/en_split.json
data/manifests/tr_split.json
```

Text-bearing files belong under Git-ignored data directories. Split manifests
contain opaque IDs, labels, counts, seeds, and fingerprints only.

## Reproducible workflow

### 1. Parse authorized source files

```bash
clinical-re prepare-i2b2 \
  --concepts-dir data/raw/en/concepts \
  --relations-dir data/raw/en/relations \
  --text-dir data/raw/en/text \
  --language en \
  --output data/interim/en_all.tsv
```

### 2. Create a train/development/test manifest

```bash
clinical-re create-split-manifest \
  --training-pool data/processed/en_training_pool.tsv \
  --official-test data/processed/en_official_test.tsv \
  --dataset-name i2b2-2010-en \
  --dev-fraction 0.2 \
  --seed 42 \
  --output data/manifests/en_split.json
```

### 3. Train RAR and create ranked demonstrations

```bash
clinical-re run-rar \
  --training-pool data/processed/en_training_pool.tsv \
  --official-test data/processed/en_official_test.tsv \
  --manifest data/manifests/en_split.json \
  --language en \
  --output-dir checkpoints/rar/en
```


### 4. Run an LLM experiment

```bash
clinical-re run-inference \
  --training-pool data/processed/en_training_pool.tsv \
  --official-test data/processed/en_official_test.tsv \
  --rankings checkpoints/rar/en/rankings.jsonl \
  --provider gemini \
  --model gemini-1.5-flash-latest \
  --num-shots 5 \
  --strategy standard \
  --temperature 0.0 \
  --output results/metrics/en_gemini15_rar_5shot.jsonl
```

Dynamic self-questioning and gold-label CoT require a local JSON mapping from
demonstration ID to generated rationale via `--rationales`. 

For zero-shot static CoT, set `--num-shots 0`, use `--strategy
zs_static_cot`, and omit `--rankings`. Few-shot static CoT uses `fs_static_cot`
with a ranking file whose rows contain the same fixed random demonstration IDs
for every test example.

### 5. Train a supervised baseline

```bash
clinical-re run-baseline \
  --baseline pure \
  --training-pool data/processed/en_training_pool.tsv \
  --official-test data/processed/en_official_test.tsv \
  --manifest data/manifests/en_split.json \
  --language en \
  --output-dir checkpoints/baselines/pure/en
```

Use `--baseline bert` for the sentence-classification baseline. Both workflows
select the best development macro-F1 checkpoint and evaluate the official test
set once.

### 6. Evaluate label-only predictions

```bash
clinical-re evaluate \
  --predictions results/metrics/en_gemini15_rar_5shot.jsonl \
  --metrics results/metrics/en_gemini15_rar_5shot_metrics.json \
  --confusion-matrix results/figures/en_gemini15_rar_5shot_confusion.csv
```

Evaluation reports accuracy, micro-F1, macro-F1, all eight per-relation
statistics, and an eight-by-eight confusion matrix.

### 7. Rebuild the released result tables

```bash
clinical-re summarize-results \
  --predictions-dir results/predictions \
  --output-dir results/summary
```

The released set contains 84 validated experiments. Every file has 500 ordered
label-only records, and all experiments for a language share the same gold-label
sequence.


## Results

The table reports the best configuration for each model and language. 

| Model | English configuration | English micro-F1 | Turkish configuration | Turkish micro-F1 |
| --- | --- | ---: | --- | ---: |
| Gemini 1.5 Flash | RAR, 5-shot standard | 0.906 | RAR, 5-shot standard | 0.870 |
| Gemini 2.0 Flash | RAR, 5-shot standard | 0.906 | RAR, 5-shot Output Format CoT | 0.856 |
| GPT-4o mini | RAR, 5-shot Self-Questioning | 0.832 | Random, 5-shot Static CoT | 0.756 |
| DeepSeek V3 | RAR, 5-shot Output Format CoT | 0.918 | RAR, 5-shot Output Format CoT | 0.858 |
| Gemini 3.0 Flash | RAR, 5-shot Output Format CoT | **0.936** | RAR, 5-shot Output Format CoT | **0.908** |
| GPT-5.4 mini | RAR, 5-shot standard | 0.834 | RAR, 5-shot Output Format CoT | 0.804 |
| DeepSeek V4 Flash | RAR, 5-shot Output Format CoT | 0.892 | RAR, 5-shot standard | 0.844 |

Complete experiment-level results, per-relation metrics, and confusion matrices
are available in `results/summary/`.

## Configuration

- `config/models.yaml`: language-specific retriever, baseline, and LLM names.
- `config/rar.yaml`: pair construction, SimCSE training, and retrieval weights.
- `config/baselines.yaml`: BERT and PURE training hyperparameters.
- `config/experiments.yaml`: languages, shot counts, prompt strategies, seed,
  and inference temperature.


## Development checks

```bash
python -m pytest
ruff check .
ruff format --check .
```

All tests use synthetic text. The MIT license covers the repository code only;
it does not grant rights to the clinical dataset or derived text.
