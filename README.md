# Learning Languages Project – Language Exchange Matchmaking

This repo contains a small **language-exchange matchmaking system** (demo) and **experiment scripts**.

The easiest way for a reviewer/TA to run everything is via the **Google Colab notebook**.

---

## Option A (recommended): Run in Google Colab

1) Open the notebook in Colab:

- https://colab.research.google.com/github/shimona4321-collab/learning-languages-project/blob/main/Language_Exchange_Matching_Colab.ipynb

2) In Colab, click **Runtime → Run all**.

3) What the notebook runs:

- **Setup**: clones this repo into Colab and imports the code.
- **Interactive demo**: shows how to add/remove users, run matching rounds, and accept/reject proposals.
- **Experiments**: runs the experiment blocks and outputs the graphs.

Notes:
- The cell **"Visualize Matching Graph"** is optional and is **disabled by default** to avoid freezing on large graphs.
- If something gets stuck in Colab: use **Runtime → Restart runtime**, then run again.

---

## How to use the demo inside the notebook

Inside the notebook you can:

- **Add users** using the helper `register_user(...)`.
- **Run a round** using `lm.run_matching_round(state)`.
- **View current proposals** in the "View Proposals" cell.
- **Accept / Reject** using `lm.handle_proposal_response(state, proposal_id, accept=True/False)`.

The notebook includes ready-made cells that demonstrate these actions.

---

## Option B: Run locally (optional)

Requirements:
- Python 3.10+
- numpy, matplotlib

Run the interactive CLI demo:

```bash
python -m app.main
```

Run experiments (graphs saved under `app/plots/`):

```bash
python -m app.checking_langmatc
```

---

## Repository link

- https://github.com/shimona4321-collab/learning-languages-project
