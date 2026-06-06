# Chess Elo Predictor ♟️

An end-to-end machine learning project that predicts a chess player's Elo rating class from raw game data and Stockfish engine evaluations. Given a set of chess games in PGN format, the pipeline extracts behavioural features (centipawn loss, blunders, game chaos, captures) and trains a multi-class XGBoost classifier to categorise each player into one of three skill brackets.

**Current performance: ~58.4% bracket accuracy on the held-out test set.**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Elo Brackets](#elo-brackets)
- [Features Extracted](#features-extracted)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset Access](#dataset-access)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Project Overview

Chess players at different skill levels make very different types of mistakes. A beginner blunders pieces frequently; an intermediate player rarely hangs pieces but may misjudge positional plans; a grandmaster operates with near-engine precision. This project quantifies those differences by:

1. Parsing raw PGN game files to extract board-level statistics per player.
2. Joining Stockfish centipawn evaluations (pre-computed) to measure move quality.
3. Engineering features that capture blunder rate, game volatility, and overall positional chaos.
4. Training an XGBoost classifier to predict the player's Elo bracket.

---

## Elo Brackets

The target variable is a 3-class label based on the White player's Elo:

| Class | Label | Elo Range |
|-------|-------|-----------|
| 0 | Beginner / Club | < 1800 |
| 1 | Intermediate | 1800 – 2199 |
| 2 | Expert / Master | ≥ 2200 |

---

## Features Extracted

For each game, the following features are computed **per colour (White / Black)**:

| Feature | Description |
|---|---|
| `avg_cpl` | Average centipawn loss — how many centipawns the player loses per move on average |
| `max_swing` | Largest single-move position swing (in centipawns) |
| `chaos` | Sum of all position swings — measures overall game turbulence |
| `blunders` | Number of moves with a centipawn loss > 300 |
| `W_Caps / B_Caps` | Total captures made by each side |
| `NumMoves` | Total half-moves in the game |
| `W_Volatility` | Blunder count normalised by game length (`blunders / (moves + 1)`) |
| `B_Volatility` | Same metric for Black |

Centipawn scores are clipped to ±1500 before processing to reduce noise from forced-mate sequences.

---

## Repository Structure

```
ChessElo/
├── ChessElo.ipynb              # Main notebook: feature extraction → training → evaluation
├── requirements                # Python dependencies (pip install -r requirements)
├── data.pgn                    # Raw PGN game file (download from Kaggle — see below)
├── stockfish.csv               # Pre-computed Stockfish move scores (download from Kaggle)
├── ultimate_features_full.csv  # Cached feature table (auto-generated on first run)
└── finding-elo/                # Original Kaggle competition data archives
    ├── data.pgn.zip
    ├── data_uci.pgn.zip
    ├── sampleSubmission.csv.zip
    └── stockfish.csv.zip
```

> **Note:** `data.pgn` (~30 MB) and `stockfish.csv` (~15 MB) are not committed to Git. See [Dataset Access](#dataset-access) for instructions.

---

## Environment Setup

Python 3.12 and conda are recommended.

```bash
# 1. Create and activate a fresh conda environment
conda create -n chesselo python=3.12
conda activate chesselo

# 2. Install dependencies
pip install -r requirements
```

### Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data manipulation and CSV I/O |
| `numpy` | Numerical computations |
| `python-chess` | PGN parsing and board move logic |
| `xgboost` | Gradient-boosted classifier |
| `scikit-learn` | Train/test split, metrics, class weighting |
| `tqdm` | Progress bars during feature extraction |

---

## Dataset Access

The raw chess datasets are sourced from the Kaggle competition **[Finding Elo – Predict chess player ratings](https://www.kaggle.com/competitions/finding-elo)**.

Because the files are large, they are not stored in this repository. To download them:

1. Create a free Kaggle account if you don't have one.
2. Accept the competition rules on the competition page linked above.
3. Go to the **Data** tab and download the following files:
   - `data.pgn` — raw game records
   - `stockfish.csv` — pre-computed engine evaluations per game
4. Place both files directly inside the `ChessElo/` project folder.

The expected layout after downloading:

```
ChessElo/
├── data.pgn
├── stockfish.csv
└── ChessElo.ipynb
```

---

## Usage

```bash
# 1. Make sure your environment is active
conda activate chesselo

# 2. Launch Jupyter
jupyter notebook

# 3. Open ChessElo/ChessElo.ipynb and run all cells in order
```

### What each notebook part does

| Part | Cells | Description |
|------|-------|-------------|
| **Part 1 – Feature Extraction** | 3–4 | Parses `data.pgn`, joins `stockfish.csv`, computes all features, saves `ultimate_features_full.csv` (skipped if cache exists) |
| **Part 2 – Data Preparation** | 5–6 | Assigns bracket labels, engineers volatility features, splits into train/test sets with balanced class weights |
| **Part 3 – Model Training** | 7–8 | Trains an XGBoost multi-class classifier with early stopping |
| **Part 4 – Evaluation** | 9–10 | Reports accuracy, per-class precision/recall/F1, and confusion matrix |

> Feature extraction over 25,000 games takes several minutes on first run. The result is cached to `ultimate_features_full.csv` so subsequent runs skip this step.

---

## Model Details

```
XGBClassifier(
    n_estimators       = 1000,
    learning_rate      = 0.05,
    max_depth          = 6,
    subsample          = 0.8,
    colsample_bytree   = 0.8,
    objective          = 'multi:softprob',
    num_class          = 3,
    early_stopping_rounds = 50
)
```

- **Class imbalance** is handled via `compute_sample_weight(class_weight='balanced')` from scikit-learn.
- **Early stopping** monitors the test-set loss and halts training if there is no improvement for 50 consecutive rounds.
- **Train / test split:** 80% / 20%, `random_state=42`.

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **58.4%** |

A per-class classification report and confusion matrix are printed at the end of the notebook.

---

## Future Improvements

- **More features:** opening classification, move-time variance (if available), endgame-specific metrics, or game phase breakdown (opening / middlegame / endgame CPL).
- **Alternative models:** Random Forest, LightGBM, SVM, or a sequence model (LSTM / Transformer) operating directly on move embeddings.
- **Hyperparameter tuning:** Optuna or grid search over XGBoost parameters.
- **Predict both players:** the current model only predicts White's bracket; extending it to Black would double the training signal.
- **Regression instead of classification:** predict raw Elo as a continuous value and report MAE.
- **UCI data:** the `data_uci.pgn` file is available but not yet used; it could be incorporated to increase dataset size.
