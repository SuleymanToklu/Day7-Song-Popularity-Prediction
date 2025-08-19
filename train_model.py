import pandas as pd
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_training_pipeline():
    """
    Loads Spotify song data, cleans it thoroughly to ensure all features are numeric,
    trains a regression model, and saves the artifacts.
    """
    print("--- Training Pipeline Started ---")

    try:
        df = pd.read_csv("SpotifyFeatures.csv")
    except FileNotFoundError:
        print("HATA: 'SpotifyFeatures.csv' bulunamadı.")
        return

    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    target = 'popularity'
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=features + [target], inplace=True)

    X = df[features]
    y = df[target]
    
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=7, learning_rate=0.1)
    model.fit(X, y)

    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    print("--- Training Pipeline Completed Successfully! ---")
    print("Yeni 'model.pkl' ve 'model_features.pkl' dosyaları kaydedildi.")

if __name__ == "__main__":
    run_training_pipeline()
