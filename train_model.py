"""
train_model.py
--------------
Run this script ONCE to train the KMeans model on the dataset
and save kmeans_model.pkl and scaler.pkl for use in the Streamlit app.

Usage:
    python train_model.py
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

#   1. Load Data                                
print("Loading dataset...")
df = pd.read_csv("Final_Amazon_Music_Project.csv")
print(f"  Rows: {len(df)}")

#   2. Select the EXACT 10 features used in the notebook           ─
FEATURES = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'duration_ms', 'valence', 'tempo'
]

X = df[FEATURES].copy()

#   3. Scale                                  
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#   4. Train KMeans (k=5, same as notebook)                  ─
print("Training KMeans (k=5)...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_scaled)
print(f"  Inertia: {kmeans.inertia_:.2f}")

#   5. Save artifacts                             ─
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
print("✅ Saved: scaler.pkl, kmeans_model.pkl")
