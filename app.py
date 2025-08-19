import streamlit as st
import pandas as pd
import joblib
import requests
import base64
import time
import warnings
import os

warnings.filterwarnings("ignore")

texts = {
    'page_title': {'TR': 'Şarkı Popülerliği Tahmini', 'EN': 'Song Popularity Prediction'},
    'main_title': {'TR': '🎵 Şarkı Popülerliği Tahmincisi', 'EN': '🎵 Song Popularity Predictor'},
    'model_error': {'TR': "Gerekli model dosyaları bulunamadı. Lütfen önce train_model.py script'ini çalıştırdığınızdan emin olun.", 'EN': "Required model files not found. Please make sure you have run the `train_model.py` script first."},
    'language_label': {'TR': 'Dil / Language', 'EN': 'Language / Dil'},
    'sidebar_api_status': {'TR': "Spotify API Durumu", 'EN': "Spotify API Status"},
    'sidebar_api_success': {'TR': "Bağlantı hazır! ✅", 'EN': "Connection ready! ✅"},
    'sidebar_api_error': {'TR': "API bilgileri bulunamadı!", 'EN': "Spotify API credentials not found!"},
    'tab1_title': {'TR': "🎤 Spotify'dan Şarkı Ara", 'EN': "🎤 Search Song on Spotify"},
    'tab2_title': {'TR': "🎯 Proje Detayları", 'EN': "🎯 Project Details"},
    'tab1_header': {'TR': "Spotify'da Şarkı Arayarak Popülerlik Tahmin Et", 'EN': "Predict Popularity by Searching on Spotify"},
    'search_form_label': {'TR': "Şarkı Adı ve/veya Sanatçı", 'EN': "Song Name and/or Artist"},
    'search_button': {'TR': 'Ara', 'EN': 'Search'},
    'search_results_header': {'TR': "Arama Sonuçları", 'EN': "Search Results"},
    'artist_label': {'TR': "Sanatçı", 'EN': "Artist"},
    'predict_button': {'TR': "Popülerliği Tahmin Et", 'EN': "Predict Popularity"},
    'prediction_header': {'TR': "🔮 '{track_name}' için Tahmin Sonucu", 'EN': "🔮 Prediction Result for '{track_name}'"},
    'metric_prediction': {'TR': "Modelin Tahmini Popülerlik Puanı", 'EN': "Model's Predicted Popularity Score"},
    'metric_real': {'TR': "Gerçek Spotify Popülerlik Puanı", 'EN': "Actual Spotify Popularity Score"},
    'not_in_dataset_warning': {'TR': "Bu şarkı modelimizin eğitildiği veri setinde bulunamadı. Bu yüzden özel bir tahmin yapılamıyor.", 'EN': "This song was not found in the dataset our model was trained on. Therefore, a custom prediction cannot be made."},
    'close_button': {'TR': 'Kapat', 'EN': 'Close'},
    'tab2_header': {'TR': "Projenin Amacı ve Teknik Detaylar", 'EN': "Project Goal and Technical Details"},
    'tab2_text': {'TR': """Bu projenin amacı, bir şarkının Spotify'daki popülerliğini, Spotify API tarafından sağlanan ses özelliklerine göre tahmin etmektir. \n- **Model:** `XGBoost Regressor`\n- **Veri Seti:** Spotify Features (Kaggle) & Live Spotify API""", 'EN': """The goal of this project is to predict the popularity of a song on Spotify based on its audio features provided by the Spotify API. \n- **Model:** `XGBoost Regressor`\n- **Dataset:** Spotify Features (Kaggle) & Live Spotify API"""},
    'expander_title': {'TR': "🎵 Ses Özellikleri Ne Anlama Geliyor?", 'EN': "🎵 What Do the Audio Features Mean?"},
    'expander_text': {'TR': """- **Dans Edilebilirlik (Danceability):** Dans etmeye ne kadar uygun olduğunu açıklar.\n- **Enerji (Energy):** Yoğunluk ve aktivitenin algısal bir ölçüsüdür.\n- **Gürültü (Loudness):** Genel ses yüksekliği (desibel - dB).\n- **Akustiklik (Acousticness):** Parçanın akustik olup olmadığının bir ölçüsü.\n- **Enstrümantallik (Instrumentalness):** Parçanın vokal içerip içermediğini tahmin eder.\n- **Pozitiflik (Valence):** Müziksel pozitifliği (mutlu, neşeli) açıklar.\n- **Tempo:** Dakikadaki vuruş sayısı (BPM).""", 'EN': """- **Danceability:** Describes how suitable a track is for dancing.\n- **Energy:** A perceptual measure of intensity and activity.\n- **Loudness:** The overall loudness of a track in decibels (dB).\n- **Acousticness:** A measure of whether the track is acoustic.\n- **Instrumentalness:** Predicts whether a track contains no vocals.\n- **Valence:** Describes the musical positiveness (e.g., happy, cheerful) conveyed by a track.\n- **Tempo:** The overall estimated tempo of a track in beats per minute (BPM)."""}
}

st.session_state.setdefault('tracks', [])
st.session_state.setdefault('selected_track', None)
st.session_state.setdefault('lang', 'TR')
st.session_state.setdefault('access_token', None)
st.session_state.setdefault('token_expires', 0)

st.sidebar.selectbox(label=texts['language_label'][st.session_state.lang], options=['TR', 'EN'], key='lang')
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
        return df
    except FileNotFoundError:
        return None

model, model_features = load_model_and_features()
local_df = load_local_dataset()

st.title(texts['main_title'][lang])

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

st.sidebar.subheader(texts['sidebar_api_status'][lang])
api_ready = False
try:
    cid = st.secrets["SPOTIPY_CLIENT_ID"]
    csecret = st.secrets["SPOTIPY_CLIENT_SECRET"]
    if not is_token_valid():
        get_spotify_token(cid, csecret)
    st.sidebar.success(texts['sidebar_api_success'][lang])
    api_ready = True
except Exception:
    st.sidebar.error("API bağlantısı kurulamadı. Arama özelliği çalışmayabilir.")

tab1, tab2 = st.tabs([f"🎤 **{texts['tab1_title'][lang]}**", f"🎯 **{texts['tab2_title'][lang]}**"])

with tab1:
    st.header(texts['tab1_header'][lang])
    
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
                st.session_state.selected_track = None
            except Exception as e:
                st.toast(f"Arama Hatası: {e}", icon='🚨')

        if st.session_state.selected_track:
            track = st.session_state.selected_track
            track_name = track['name']
            artist_name = track['artists'][0]['name']

            with st.container(border=True):
                st.subheader(texts['prediction_header'][lang].format(track_name=track_name))
                
                match = local_df[
                    (local_df['track_name'].str.lower() == track_name.lower()) &
                    (local_df['artist_name'].str.lower() == artist_name.lower())
                ]

                if not match.empty:
                    song_features = match.iloc[0]
                    input_df = pd.DataFrame([song_features[model_features]])
                    prediction = model.predict(input_df)
                    popularity_score = int(prediction[0])

                    col_pred, col_real = st.columns(2)
                    col_pred.metric(label=texts['metric_prediction'][lang], value=popularity_score)
                    col_pred.progress(popularity_score)
                    col_real.metric(label=texts['metric_real'][lang], value=track['popularity'])
                    col_real.progress(track['popularity'])
                else:
                    st.warning(texts['not_in_dataset_warning'][lang])
                    st.metric(label=texts['metric_real'][lang], value=track['popularity'])
                    st.progress(track['popularity'])

                if st.button(texts['close_button'][lang]):
                    st.session_state.selected_track = None
                    st.rerun()

        elif st.session_state.tracks:
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
    st.header(texts['tab2_header'][lang])
    st.write(texts['tab2_text'][lang])
    with st.expander(texts['expander_title'][lang]):
        st.markdown(texts['expander_text'][lang])
