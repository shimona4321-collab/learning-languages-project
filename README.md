# Learning Languages Project – Language-Exchange Matchmaking

This repository contains a small end-to-end system for matching Hebrew ↔ English conversation partners.

**Course submission entrypoint:**
- `Language_Exchange_Matching_Colab.ipynb` (runs in Google Colab)

---

## Option A: Run everything in Colab (recommended)

1) Open the notebook in Colab:

`https://colab.research.google.com/github/shimona4321-collab/learning-languages-project/blob/main/Language_Exchange_Matching_Colab.ipynb`

2) In Colab click: **Runtime → Run all**.

Running all cells will execute:
- A short non-interactive demo (few users)
- Experiments that produce the result plots

### Notes
- The notebook is **quiet by default** (no per-round logs).
- If you want to see internal per-round logs, set in the setup cell:
  `os.environ["LANGMATCH_VERBOSE"] = "1"`

---

## Option B: Interactive demo inside the notebook (optional)

The notebook also includes an **interactive** section.

- Find the section named **Interactive notebook demo**.
- Set:
  `RUN_INTERACTIVE_DEMO = True`
- Run only that section’s cells (do *not* use “Run all”).

In interactive mode you can:
- Add a user
- Remove a user
- View current users
- Run a matching round and inspect proposals

---

## Local run (optional)

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Run the interactive CLI:

```bash
python -m app.main
```

3) Run the experiments script:

```bash
python -m app.checking_langmatc
```

You can keep experiments quiet by setting:

```bash
set LANGMATCH_VERBOSE=0
```
(on Windows PowerShell use `$env:LANGMATCH_VERBOSE="0"`).
