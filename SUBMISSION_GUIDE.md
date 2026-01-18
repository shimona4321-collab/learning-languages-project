# Submission Guide (Quick)

This repository includes:
- `Language_Exchange_Matching_Colab.ipynb` – the **main Colab notebook** (demo + experiments)
- `README.md` – local run instructions


## Recommended: GitHub + Colab (easiest for the instructor)

### 1) Create a GitHub repository
Push this project so that the **repo root** contains:
- `Language_Exchange_Matching_Colab.ipynb`
- `requirements.txt`
- `README.md`
- `app/` (the Python package)

### 2) Open the notebook in Colab
In Colab:
- File → Open notebook → **GitHub** tab
- Paste your repo URL and open `Language_Exchange_Matching_Colab.ipynb`

### 3) Set the repo URL once
In **Section 1.1** (Auto-setup), set:
- `REPO_URL = "https://github.com/<username>/<repo>.git"`
Then run:
- Runtime → **Run all**

### 4) Share links to the instructor
Send:
1) **Colab link** (Share → "Anyone with the link" can view/run)
2) **GitHub repo** link (source code + README)


## Alternative: share a ZIP (local run)
If you prefer not to use GitHub:
1) Send the ZIP file to the instructor (or upload to Google Drive and share).
2) Instructor runs locally:

```bash
pip install -r requirements.txt
python -m app.main
python -m app.checking_langmatc
```
