# Language Exchange Matchmaking System

## Quick start (for the instructor)

### Option A: Run everything in Google Colab (recommended)
1. Open the notebook: `Language_Exchange_Matching_Colab.ipynb` (from this repo) in Colab.
2. In Colab: **Runtime → Run all**.

By default, **Run all** executes:
- A short *non-interactive demo* (predefined users + a few rounds)
- The *experiments* that generate the plots

It does **not** start the interactive CLI unless you enable it explicitly.

### Optional: Interactive demo inside Colab
In the notebook, find the cell **"Interactive CLI (optional)"** and set:
`RUN_INTERACTIVE = True`, then run **only that cell**.

> Note: The CLI is designed for a terminal. In Colab it runs inside the output area; if you get stuck, use **Runtime → Restart runtime**.

### Option B: Run locally
```bash
pip install -r requirements.txt
python -m app.main              # interactive CLI (Admin/User)
python -m app.checking_langmatc  # experiments + plots
```

A language exchange matchmaking system using AI (LinUCB contextual bandits) for optimal partner matching.

## Project Overview

This project implements an intelligent matchmaking system for language exchange partners. It uses:
- **LinUCB (Linear Upper Confidence Bound)** - A contextual bandit algorithm for learning user preferences
- **Bipartite Graph Matching** - For optimal pairing of users
- **Personalization** - Per-user bandits that learn individual preferences over time

## Requirements

- Python 3.8+
- numpy
- matplotlib
- networkx
- scipy (optional, for Hungarian algorithm optimization)

## Installation

### Option 1: Local Installation

```bash
# Clone or download the project
cd Learning_languages_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Google Colab

1. Upload the project files to Google Colab or connect to Google Drive
2. Run the following in a Colab cell:

```python
!pip install numpy matplotlib networkx scipy
```

## Running the Application

### Main Application (Interactive CLI)

Recommended (most robust):

```bash
python -m app.main
```

Alternative (also works):

```bash
python app/main.py
```

The application has two modes:
1. **Admin Mode** - Manage users, run matching rounds, configure parameters
2. **User Mode** - Individual user interaction with the system

### Admin Mode Options:
- Register new users
- Delete users
- Bulk-generate random users (for testing)
- Run matching rounds
- View bipartite graph visualization
- Inspect bandit parameters
- Configure scoring weights

### User Mode Options:
- View current proposal
- Accept/Reject proposals
- View match history

## Running Experiments

The experiments validate the AI matching algorithm performance.

Recommended (most robust):

```bash
python -m app.checking_langmatc
```

Alternative (also works):

```bash
python app/checking_langmatc.py
```

### Available Experiments:

| # | Experiment | Description |
|---|-----------|-------------|
| 1 | Random vs Bandit | Compares random matching vs AI-based matching using Preference Alignment Score (PAS) |
| 2 | Exploration ON vs OFF | Tests the impact of exploration (alpha parameter) on novelty rate |
| 3 | Personalization ON vs OFF | Compares global-only vs personalized matching |

### Experiment Parameters:
Each experiment allows you to configure:
- `seed` - Random seed for reproducibility
- `n_hebrew` / `n_english` - Number of users per language group
- `rounds` - Number of matching rounds to run
- `threshold` - Accept/reject threshold for simulated user behavior

### Output:
- Graphs are saved to `app/plots/` directory
- Summary statistics printed to console

## Project Structure

```
Learning_languages_project/
├── app/
│   ├── main.py              # Main entry point
│   ├── langmatch.py         # Compatibility layer / re-exports
│   ├── config.py            # Configuration constants
│   ├── checking_langmatc.py # Experiments runner
│   ├── bandit/              # LinUCB implementation
│   │   ├── linucb.py        # Bandit algorithms
│   │   └── features.py      # Feature extraction
│   ├── matching/            # Matching logic
│   │   ├── matcher.py       # Bipartite matching
│   │   ├── scoring.py       # Score computation
│   │   └── cooldown.py      # Cooldown management
│   ├── models/              # Data models
│   │   ├── user.py          # User model
│   │   ├── state.py         # Application state
│   │   └── proposal.py      # Proposal model
│   ├── persistence/         # State persistence
│   │   └── storage.py       # JSON storage
│   ├── ui/                  # User interface
│   │   ├── admin.py         # Admin mode UI
│   │   ├── user_mode.py     # User mode UI
│   │   └── visualization.py # Graph visualization
│   └── plots/               # Saved experiment graphs
├── requirements.txt
└── README.md
```

## Key Algorithm Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ALPHA_PERSONAL` | 1.0 | Exploration parameter for personal bandits |
| `ALPHA_GLOBAL` | 1.0 | Exploration parameter for global bandit |
| `EPS_WAITING` | 0.1 | Waiting-time fairness multiplier |
| `PERSONAL_RECENCY_GAMMA` | 0.95 | Decay factor for recency-weighted bandit |

## State Persistence

The application saves state to `langmatch_state.json` (override via the environment variable `LANGMATCH_STATE_FILE`) including:
- All registered users
- Bandit parameters (A matrix, b vector)
- Active proposals
- Pair cooldowns
- Scoring weights

---

## For Course Submission (Google Colab)

Recommended: use the included notebook **Language_Exchange_Matching_Colab.ipynb**.

**Best workflow (for a 1-click experience for the instructor):**
1) Upload this project to GitHub.
2) Open the notebook in Colab.
3) In the first setup cell, set `REPO_URL` to your GitHub repo, then choose **Runtime → Run all**.

**Alternative (no GitHub):**
- Upload the project ZIP to Colab, unzip it, and rerun the setup cell.

Example commands (inside Colab):

```python
!pip -q install -r requirements.txt
!python -m app.checking_langmatc
```


---

## Author

Project for AI for Social Good course (#55982)
Hebrew University - Business School
Semester A 2025-2026