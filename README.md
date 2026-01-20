# Language Exchange Matchmaking System

This project matches language-exchange partners using:
- **Weighted scoring** over language / availability / shared topics
- **Contextual bandits (LinUCB)** for exploration vs exploitation
- **Fairness / diversity boosting** (to avoid repeatedly offering the same strong users)

## Option A : Run everything in Colab (automated demo + experiments)

1. Open the notebook:
   - `Language_Exchange_Matching_Colab.ipynb` (from this repo)

2. In Colab:
   - `Runtime` → `Run all`

What you get when you run-all:
- A **small non-interactive demo** (creates a few synthetic users and runs a matching round)
- **Experiments + graphs** (bandit vs baselines, exploration on/off, fairness effect)
- No manual inputs are required in run-all.

> Tip: If Colab is slow/stuck, use `Runtime → Restart runtime` and then `Runtime → Run all`.

## Option B: Interactive demo (optional)

The notebook also includes an **optional interactive demo** (add/remove users, run rounds, accept/reject offers).
This is **disabled by default** so that `Run all` never gets stuck waiting for input.

To enable it:
1. Find the cell titled **“OPTIONAL: Interactive Demo (Admin/User)”**
2. Set `RUN_INTERACTIVE = True`
3. Run that cell (it will start asking for inputs)


## Repository contents

- `Language_Exchange_Matching_Colab.ipynb` — the Colab notebook (Option A + Option B)
- `app/` — the project code (matching, bandit, simulation, UI)
- `requirements.txt` — dependencies
