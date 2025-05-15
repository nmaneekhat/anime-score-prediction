# Anime Score Prediction Project 🎯

This project uses a machine learning model to predict anime scores based on features like genres, episode count, popularity, and release year. It was built using Python and scikit-learn as part of a Machine Learning course (CS379).

## 📁 Dataset

- **Source**: [Anime Dataset 2025 on Kaggle](https://www.kaggle.com/datasets/dyaquo/anime-dataset-2025)
- The dataset includes:
  - `mean_score`: The average rating of the anime
  - `genres`: A comma-separated list of genres
  - `episodes`: Total number of episodes
  - `popularity`: Popularity score
  - `start`: Original air date

## 🧠 Model

- **Algorithm**: Random Forest Regressor (from `sklearn.ensemble`)
- **Why Random Forest?** It performs well with mixed feature types and handles non-linear relationships effectively.

## 🔧 Preprocessing Steps

- Cleaned and standardized column names
- One-hot encoded multi-label `genres`
- Converted `start` to release year
- Removed missing values

## 📈 Results

- Evaluated using:
  - Mean Squared Error (MSE)
  - R² Score
- Shows actual vs. predicted scores for a sample of test data

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/nmaneekhat/anime-score-prediction.git
   cd anime-score-prediction
