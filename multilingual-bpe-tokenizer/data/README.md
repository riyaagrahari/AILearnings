# Data folder — expected structure

This project does **not** ship, download, or generate any training or
evaluation data. You provide real text yourself, in this exact layout:

```
data/
├── raw/            # training corpus — read by scripts/train_tokenizer.py
│   ├── en/*.txt
│   ├── hi/*.txt
│   ├── te/*.txt
│   └── ta/*.txt
└── eval/           # held-out evaluation text — read by scripts/evaluate_tokenizer.py
    ├── en/*.txt
    ├── hi/*.txt
    ├── te/*.txt
    └── ta/*.txt
```

`en` = English, `hi` = Hindi, `te` = Telugu, `ta` = Tamil — these four
subfolder names are the defaults `scripts/train_tokenizer.py` and
`scripts/evaluate_tokenizer.py` look for (override with `--languages` if
you use different codes or a subset).

## File format

- **Plain UTF-8 text** (a leading BOM is tolerated and stripped). One or
  more `.txt` files per language folder — file names don't matter, every
  `*.txt` file directly inside a language folder is read.
- Content can be plain prose, or Wikipedia-style markup (`{{templates}}`,
  `[[links]]`, `<ref>` tags, etc.) — `bpe.preprocess.clean_text` runs on
  every file automatically and strips markup harmlessly either way.
- No minimum file count or size is enforced, but **fertility and
  cross-language balance depend directly on how much real text you
  provide per language** — a handful of sentences will train and run
  end-to-end (useful for smoke-testing the pipeline) but won't produce
  meaningful fertility numbers. For results resembling the assignment's
  targets (English fertility ≤ 1.2, balanced fertility across all four
  languages), each language needs a substantial, representative corpus.
- `data/eval/<lang>/` must contain text **disjoint from** `data/raw/<lang>/`
  (held out, not seen during training) — otherwise fertility/UNK-rate/
  compression numbers measure memorization, not generalization.

## What happens if a folder is empty

- `scripts/train_tokenizer.py` / `scripts/evaluate_tokenizer.py` print a
  warning naming any language folder with no usable `.txt` content, and
  continue with whatever languages *do* have data.
- If **every** language folder is empty, both scripts exit with an error
  and write nothing — they never fall back to synthetic or downloaded
  data.

## Where to get real text (your responsibility)

Any UTF-8 text you have rights to use works: Wikipedia dumps/exports,
public-domain corpora, your own writing, etc. This repository does not
bundle or fetch any of it.
