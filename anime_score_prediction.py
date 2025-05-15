# anime_score_prediction.py
# Date: May 15, 2025
# Description:
# This script builds a machine learning model to predict anime scores using a Random Forest Regressor.
# It loads and preprocesses the dataset, encodes categorical features, trains the model, evaluates its performance,
# and displays sample predictions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from an Excel file
file_path = r"Please replace this with file path of 'myanilist.xlsx"
anime_df = pd.read_excel(file_path)

# Display initial sample and original column names
print("First 5 rows of the dataset:")
print(anime_df.head())
print("\nOriginal columns in the dataset:")
print(list(anime_df.columns))

# Clean and standardize column names
anime_df.columns = anime_df.columns.str.strip().str.lower().str.replace(' ', '_')

print("\nCleaned columns in the dataset:")
print([repr(col) for col in anime_df.columns])

# Select relevant columns and remove rows with missing values
selected_columns = ['mean_score', 'genres', 'episodes', 'popularity', 'start']
anime_df = anime_df[selected_columns].dropna()

print("\nData shape after selecting columns and dropping missing values:")
print(anime_df.shape)
print("\nFirst 5 rows of filtered data:")
print(anime_df.head())

# One-hot encode the 'genres' column (multi-label)
anime_df['genres'] = anime_df['genres'].apply(lambda x: [genre.strip() for genre in x.split(',')])
all_genres = set()
for genres_list in anime_df['genres']:
    all_genres.update(genres_list)
for genre in all_genres:
    anime_df[f'genre_{genre}'] = anime_df['genres'].apply(lambda x: 1 if genre in x else 0)
anime_df = anime_df.drop(columns=['genres'])

# Convert 'start' to year (integer format)
anime_df['start'] = pd.to_datetime(anime_df['start'], errors='coerce').dt.year
anime_df = anime_df.dropna(subset=['start']).astype({'start': 'int'})

print("\nColumns after genre encoding and processing 'start':")
print(anime_df.columns)

# Define features and target label
features = anime_df.drop(columns=['mean_score'])
labels = anime_df['mean_score']

print("\nFeatures shape:", features.shape)
print("Labels shape:", labels.shape)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Display a sample of actual vs. predicted values
predictions_df = pd.DataFrame({
    'Actual Score': y_test[:10].round(2),
    'Predicted Score': y_pred[:10].round(2)
})
print("\nSample of Actual vs Predicted Scores:")
print(predictions_df)
