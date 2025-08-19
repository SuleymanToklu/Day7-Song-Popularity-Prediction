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
    print("--- EĞİTİM SÜRECİ BAŞLADI ---")

    try:
        df = pd.read_csv("SpotifyFeatures.csv")
        print("-> 'SpotifyFeatures.csv' başarıyla yüklendi.")
    except FileNotFoundError:
        print("HATA: 'SpotifyFeatures.csv' bulunamadı. Lütfen dosyanın doğru klasörde olduğundan emin olun.")
        return

    features = [
        'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms'
    ]
    target = 'popularity'
    
    print(f"-> Model {len(features)} özellik ile eğitilecek: {features}")
    
    for col in features + [target]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    initial_rows = len(df)
    df.dropna(subset=features + [target], inplace=True)
    cleaned_rows = len(df)
    print(f"-> Eksik veriler temizlendi. {initial_rows - cleaned_rows} satır silindi. Kalan satır sayısı: {cleaned_rows}")

    if cleaned_rows == 0:
        print("HATA: Temizlik sonrası hiç veri kalmadı. Lütfen CSV dosyanızı kontrol edin.")
        return

    X = df[features]
    y = df[target]
    
    print("-> XGBoost modeli eğitiliyor...")
    model = XGBRegressor(random_state=42, n_estimators=100, max_depth=7, learning_rate=0.1)
    model.fit(X, y)
    print("-> Model başarıyla eğitildi.")

    joblib.dump(model, 'model.pkl')
    joblib.dump(features, 'model_features.pkl')
    
    print("\n--- EĞİTİM SÜRECİ BAŞARIYLA TAMAMLANDI! ---")
    print("-> Yeni 'model.pkl' ve 'model_features.pkl' dosyaları kaydedildi.")

if __name__ == "__main__":
    run_training_pipeline()