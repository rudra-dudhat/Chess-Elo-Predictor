# Chess Elo Predictor ♟️

An end-to-end machine learning project that predicts a chess player's Elo rating class using raw game data and engine evaluations.

## 📊 Overview
This project parses chess games and extracts complex features (such as Centipawn Loss, blunders, and game swing) to categorize player strength. The current implementation relies on a Multi-Class XGBoost Model, achieving an Overall Bracket Accuracy of 58.4%.

## 📂 Repository Structure
* `ChessElo/ChessElo.ipynb`: The main Jupyter Notebook containing data preprocessing, feature extraction, and model training[cite: 1].
* `requirements`: The list of Python dependencies required to run the environment.

## ⚙️ Environment Setup & Installation
To run this project locally, you need to set up the appropriate Python environment. Run the following commands in your terminal:

```bash
# 1. Create a new conda environment with Python 3.12
conda create -n venv python=3.12

# 2. Activate the environment
conda activate venv

# 3. Install the required dependencies
pip install -r requirements


## 💾 Dataset Access
Because the raw chess datasets are too large to host on GitHub, they must be downloaded manually before running the notebook.

The datasets used in this project are from the Kaggle competition: [Finding Elo - Predict chess player ratings](https://www.kaggle.com/competitions/finding-elo).

1. Go to the link above, navigate to the **Data** tab, and download the files.
2. Extract the `ChessElo.zip` archive verbatim into your main project directory[cite: 1].
3. Ensure the structure contains `ChessElo/data.pgn` and the Stockfish evaluations (`stockfish.csv`) in the exact folder layout so the notebook can read them seamlessly[cite: 1].

## 🚀 Usage
1. Ensure your virtual environment is active (`conda activate venv`).
2. Launch Jupyter Notebook (`jupyter notebook`).
3. Open `ChessElo/ChessElo.ipynb` and run the cells sequentially to extract features and train the model[cite: 1].

## 📈 Future Improvements
* Test additional machine learning architectures (e.g., Decision Trees, Support Vector Machines, or Convolutional Neural Networks).
* Fine-tune the XGBoost hyperparameters to improve classification accuracy beyond the current baseline.