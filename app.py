import streamlit as st
import pandas as pd
import joblib
import requests
import base64
import time
import warnings
import os
import re

warnings.filterwarnings("ignore")

texts = {
    'page_title': {'TR': 'Şarkı Popülerliği Tahmini', 'EN': 'Song Popularity Prediction'},
    'main_title': {'TR': '🎵 Şarkı Popülerliği Tahmincisi', 'EN': '🎵 Song Popularity Predictor'},
    'model_error': {'TR': "Gerekli model dosyaları bulunamadı. Lütfen önce train_model.py script'ini çalıştırdığınızdan emin olun.", 'EN': "Required model files not found. Please make sure you have run the `train_model.py` script first."},
    'language_label': {'TR': 'Dil', 'EN': 'Language'},
    'api_status_success': {'TR': "Spotify API bağlantısı başarılı!", 'EN': "Spotify API connection successful!"},
    'api_status_error': {'TR': "API bağlantısı kurulamadı. Arama özelliği çalışmayabilir.", 'EN': "Could not connect to API. Search feature may not work."},
    'tab1_title': {'TR': "🎤 Tahmin Aracı", 'EN': "🎤 Predictor"},
    'tab2_title': {'TR': "🎯 Proje Detayları", 'EN': "🎯 Project Details"},
    'suggestions_header': {'TR': "✨ Veri Setinden Öneriler", 'EN': "✨ Suggestions from Dataset"},
    'refresh_button': {'TR': "Yenile", 'EN': "Refresh"},
    'search_form_label': {'TR': "Veya Spotify'da Yeni Bir Şarkı Ara", 'EN': "Or Search for a New Song on Spotify"},
    'search_button': {'TR': 'Ara', 'EN': 'Search'},
    'search_results_header': {'TR': "Arama Sonuçları", 'EN': "Search Results"},
    'artist_label': {'TR': "Sanatçı", 'EN': "Artist"},
    'predict_button': {'TR': "Popülerliği Tahmin Et", 'EN': "Predict Popularity"},
    'prediction_header': {'TR': "🔮 '{track_name}' için Tahmin Sonucu", 'EN': "🔮 Prediction Result for '{track_name}'"},
    'metric_prediction': {'TR': "Modelin Tahmini Popülerlik Puanı", 'EN': "Model's Predicted Popularity Score"},
    'metric_real': {'TR': "Gerçek Spotify Popülerlik Puanı", 'EN': "Actual Spotify Popularity Score"},
    'close_button': {'TR': 'Kapat', 'EN': 'Close'},
    'tab2_summary_header': {'TR': "Proje Özeti", 'EN': "Project Summary"},
    'tab2_summary_text': {'TR': "Bu uygulama, bir şarkının ses özelliklerini (dans edilebilirlik, enerji, tempo vb.) kullanarak o şarkının Spotify'daki popülerlik puanını (0-100 arası) tahmin etmek için geliştirilmiş bir makine öğrenmesi projesidir. Kullanıcılar, veri setinden önerilen şarkıları analiz edebilir veya Spotify API aracılığıyla yeni şarkılar arayarak hem modelin tahminini hem de şarkının gerçek popülerlik puanını karşılaştırabilir.", 'EN': "This application is a machine learning project developed to predict a song's popularity score (0-100) on Spotify using its audio features (danceability, energy, tempo, etc.). Users can analyze suggested songs from the dataset or search for new songs via the Spotify API to compare the model's prediction with the song's actual popularity score."},
    'tab2_tech_header': {'TR': "Teknik Detaylar", 'EN': "Technical Details"},
    'tab2_tech_text': {'TR': """
- **Model:** XGBoost Regressor
- **Kütüphaneler:** Streamlit, Pandas, Scikit-learn, XGBoost, Requests
- **Veri Kaynağı:** [Ultimate Spotify Tracks DB (Kaggle)](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) & Canlı Spotify Web API""", 'EN': """
- **Model:** XGBoost Regressor
- **Libraries:** Streamlit, Pandas, Scikit-learn, XGBoost, Requests
- **Data Source:** [Ultimate Spotify Tracks DB (Kaggle)](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db) & Live Spotify Web API"""},
    'tab2_how_header': {'TR': "Nasıl Çalışır?", 'EN': "How It Works?"},
    'tab2_how_text': {'TR': """
1.  **Model Eğitimi:** Proje, `train_model.py` scripti ile `SpotifyFeatures.csv` veri setini kullanarak bir XGBoost regresyon modelini eğitir ve `model.pkl` olarak kaydeder.
2.  **Veri Yükleme:** Streamlit uygulaması, bu eğitilmiş modeli ve tahmin için kullanılacak yerel veri setini başlangıçta yükler.
3.  **Tahminleme:**
    - **Veri Seti:** Veri setindeki şarkıların özellikleri doğrudan modele verilir.
    - **API Araması:** Spotify API'den aranan bir şarkının ses özellikleri (`audio features`) anlık olarak çekilir ve model bu canlı veri ile tahmin yapar.
4.  **Sonuç Gösterimi:** Modelin tahmini ve şarkının Spotify'daki gerçek popülerlik puanı karşılaştırmalı olarak kullanıcıya sunulur.""", 'EN': """
1.  **Model Training:** The project trains an XGBoost regression model using the `SpotifyFeatures.csv` dataset with the `train_model.py` script and saves it as `model.pkl`.
2.  **Data Loading:** The Streamlit application loads this pre-trained model and the local dataset for predictions at startup.
3.  **Prediction:**
    - **From Dataset:** Features of songs from the dataset are directly fed into the model.
    - **API Search:** Audio features of a song searched via the Spotify API are fetched in real-time, and the model predicts using this live data.
4.  **Result Display:** The model's prediction and the song's actual popularity score on Spotify are presented comparatively to the user."""},
    'tab2_dev_header': {'TR': "Geliştirici", 'EN': "Developer"},
    'tab2_dev_text': {'TR': "Süleyman Toklu - Isparta Uygulamalı Bilimler Üniversitesi, Bilgisayar Mühendisliği", 'EN': "Süleyman Toklu - Isparta University of Applied Sciences, Computer Engineering"}
}

st.session_state.setdefault('tracks', [])
st.session_state.setdefault('selected_track', None)
st.session_state.setdefault('lang', 'TR')
st.session_state.setdefault('access_token', None)
st.session_state.setdefault('token_expires', 0)
st.session_state.setdefault('api_status_checked', False)
st.session_state.setdefault('suggestions', None)

lang = st.session_state.lang
st.set_page_config(page_title=texts['page_title'][lang], page_icon="🎵", layout="wide")

@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('model.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, model_features
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_local_dataset():
    try:
        df = pd.read_csv('SpotifyFeatures.csv')
        model_features_list = [
            'danceability', 'energy', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        df['track_name'] = df['track_name'].astype(str).str.strip()
        df['artist_name'] = df['artist_name'].astype(str).str.strip()
        for col in model_features_list:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=model_features_list, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except FileNotFoundError:
        return None

def get_safe_sample(df, n):
    num_rows = len(df)
    sample_size = min(n, num_rows)
    if sample_size > 0:
        return df.sample(sample_size)
    return pd.DataFrame()

def display_prediction_results(predicted_score, real_score):
    col_pred, col_real = st.columns(2)
    col_pred.metric(label=texts['metric_prediction'][lang], value=predicted_score)
    col_pred.progress(predicted_score)
    col_real.metric(label=texts['metric_real'][lang], value=real_score)
    col_real.progress(real_score)

model, model_features = load_model_and_features()
local_df = load_local_dataset()

col_title, col_lang = st.columns([12, 1])
with col_title:
    st.title(texts['main_title'][lang])
with col_lang:
    st.selectbox(
        label=texts['language_label'][lang], 
        options=['TR', 'EN'], 
        key='lang',
        label_visibility="collapsed"
    )

if not model or not model_features or local_df is None:
    st.error(texts['model_error'][lang])
    st.stop()

def get_spotify_token(client_id, client_secret):
    auth_string = f"{client_id}:{client_secret}"
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = str(base64.b64encode(auth_bytes), 'utf-8')
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    result = requests.post(url, headers=headers, data=data)
    result.raise_for_status()
    json_result = result.json()
    st.session_state.access_token = json_result["access_token"]
    st.session_state.token_expires = time.time() + json_result["expires_in"]

def is_token_valid():
    return st.session_state.access_token and time.time() < st.session_state.token_expires

def spotify_search(query, token):
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 10}
    result = requests.get(url, headers=headers, params=params)
    result.raise_for_status()
    return result.json()

def get_audio_features(track_id, token):
    url = f"http://googleusercontent.com/spotify.com/2/{track_id}"
    headers = {"Authorization": f"Bearer {token}"}
    result = requests.get(url, headers=headers)
    result.raise_for_status()
    return result.json()


api_ready = False
if not st.session_state.api_status_checked:
    try:
        cid = st.secrets["SPOTIPY_CLIENT_ID"]
        csecret = st.secrets["SPOTIPY_CLIENT_SECRET"]
        if not is_token_valid():
            get_spotify_token(cid, csecret)
        st.toast(texts['api_status_success'][lang], icon='✅')
        api_ready = True
    except Exception:
        st.toast(texts['api_status_error'][lang], icon='🚨')
        api_ready = False
    st.session_state.api_status_checked = True
else:
    api_ready = is_token_valid()

if st.session_state.suggestions is None:
    st.session_state.suggestions = get_safe_sample(local_df, 5)

tab1, tab2 = st.tabs([f"🎤 **{texts['tab1_title'][lang]}**", f"🎯 **{texts['tab2_title'][lang]}**"])

with tab1:
    if st.session_state.selected_track:
        track = st.session_state.selected_track
        track_name = track['name'].strip()
        
        with st.container(border=True):
            st.subheader(texts['prediction_header'][lang].format(track_name=track_name))
            
            song_features = None
            if 'source' in track and track['source'] == 'local':
                song_features = track['features']
            else:
                try:
                    song_features = get_audio_features(track['id'], st.session_state.access_token)
                except Exception as e:
                    st.error(f"Şarkı özellikleri alınamadı: {e}")

            if song_features:
                input_df = pd.DataFrame([song_features])
                input_df = input_df[model_features]
                input_df = input_df.astype(float)
                prediction = model.predict(input_df)
                popularity_score = int(prediction[0])
                display_prediction_results(popularity_score, track['popularity'])

            if st.button(texts['close_button'][lang]):
                st.session_state.selected_track = None
                st.session_state.tracks = []
                st.rerun()
    else:
        col_header, col_button = st.columns([4, 1])
        with col_header:
            st.subheader(texts['suggestions_header'][lang])
        with col_button:
            if st.button(texts['refresh_button'][lang]):
                st.session_state.suggestions = get_safe_sample(local_df, 5)
                st.rerun()

        suggestions = st.session_state.suggestions
        if not suggestions.empty:
            for i, row in suggestions.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['track_name']}** - {row['artist_name']}")
                with col2:
                    if st.button(texts['predict_button'][lang], key=f"suggest_{i}"):
                        mock_track = {
                            'id': row['track_id'],
                            'name': row['track_name'],
                            'artists': [{'name': row['artist_name']}],
                            'popularity': row['popularity'],
                            'source': 'local',
                            'features': row.to_dict()
                        }
                        st.session_state.selected_track = mock_track
                        st.rerun()
        st.divider()

        if not api_ready:
            st.warning("API bağlantısı olmadan arama yapılamaz.")
        else:
            with st.form(key='search_form'):
                search_query = st.text_input(texts['search_form_label'][lang])
                search_button = st.form_submit_button(label=texts['search_button'][lang])

            if search_button and search_query:
                try:
                    results = spotify_search(search_query, st.session_state.access_token)
                    st.session_state.tracks = results['tracks']['items']
                except Exception as e:
                    st.toast(f"Arama Hatası: {e}", icon='🚨')

            if st.session_state.tracks:
                st.subheader(texts['search_results_header'][lang])
                for track in st.session_state.tracks:
                    col1, col2, col3 = st.columns([1, 4, 2])
                    with col1:
                        if track['album']['images']:
                            st.image(track['album']['images'][0]['url'], width=64)
                    with col2:
                        st.write(f"**{track['name']}**")
                        st.write(f"{texts['artist_label'][lang]}: {', '.join(artist['name'] for artist in track['artists'])}")
                    with col3:
                        if st.button(texts['predict_button'][lang], key=track['id']):
                            st.session_state.selected_track = track
                            st.rerun()
                st.divider()

with tab2:
    st.subheader(texts['tab2_summary_header'][lang])
    st.write(texts['tab2_summary_text'][lang])
    st.divider()
    st.subheader(texts['tab2_tech_header'][lang])
    st.write(texts['tab2_tech_text'][lang])
    st.divider()
    st.subheader(texts['tab2_how_header'][lang])
    st.write(texts['tab2_how_text'][lang])
    st.divider()
    st.subheader(texts['tab2_dev_header'][lang])
    st.write(texts['tab2_dev_text'][lang])