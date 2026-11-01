# Restricted data

This directory intentionally contains no clinical records.

Expected local-only layout:

```text
data/
├── raw/          # Authorized original i2b2/VA release
├── interim/      # Parsed text-bearing records
├── processed/    # English/Turkish experiment subsets
└── manifests/    # Public-safe ID-only split manifests
```

The text-bearing directories are ignored by Git. Do not weaken these rules.
Public examples used in tests must be synthetic and must not be paraphrases of
real patient records.
