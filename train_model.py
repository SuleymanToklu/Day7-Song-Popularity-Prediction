import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import warnings

warnings.filterwarnings("ignore")

def run_training_pipeline():
    """
    Loads Spotify song data, cleans it, trains a regression model 
    to predict popularity, and saves all necessary artifacts.
    """
    print("--- Training Pipeline Started ---")

    print("1/4 - Loading data...")
    try:
        df = pd.read_csv("SpotifyFeatures.csv")
    except FileNotFoundError:
        print("ERROR: 'SpotifyFeatures.csv' not found. Make sure it's in the same directory.")
        return

    print("2/4 - Cleaning data...")
    cols_to_convert = ['key', 'mode', 'time_signature']
    
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=cols_to_convert, inplace=True)

    df[cols_to_convert] = df[cols_to_convert].astype(int)

    print("3/4 - Preparing data and training XGBoost model...")
    features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature'
    ]
    target = 'popularity'
    
    X = df[features]
    y = df[target]
    
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=7, learning_rate=0.1)
    model.fit(X, y)

    print("4/4 - Saving artifacts...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    print("--- Training Pipeline Completed Successfully! ---")
    print("Artifacts 'model.pkl' and 'model_features.pkl' are saved.")

if __name__ == "__main__":
    run_training_pipeline()
