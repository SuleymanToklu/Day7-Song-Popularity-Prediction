import streamlit as st
import pandas as pd
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import warnings

warnings.filterwarnings("ignore")

texts = {
    'page_title': {'TR': 'Şarkı Popülerliği Tahmini', 'EN': 'Song Popularity Prediction'},
    'main_title': {'TR': '🎵 Şarkı Popülerliği Tahmincisi', 'EN': '🎵 Song Popularity Predictor'},
    'model_error': {'TR': "Gerekli model dosyaları bulunamadı. Lütfen önce `train_model.py` script'ini çalıştırdığınızdan emin olun.", 'EN': "Required model files not found. Please make sure you have run the `train_model.py` script first."},

    'language_label': {'TR': 'Dil / Language', 'EN': 'Language / Dil'},
    'sidebar_api_success': {'TR': "Spotify API bağlantısı başarılı! ✅", 'EN': "Spotify API connection successful! ✅"},
    'sidebar_api_error': {'TR': "Spotify API bilgileri bulunamadı!", 'EN': "Spotify API credentials not found!"},
    'sidebar_api_info': {'TR': "Bu uygulamayı yayınlamak için Streamlit Community Cloud'un ayarlar bölümüne API bilgilerinizi eklemeniz gerekir.", 'EN': "To deploy this app, you need to add your API credentials to the settings in Streamlit Community Cloud."},

    'tab1_title': {'TR': "🎤 Spotify'dan Şarkı Ara", 'EN': "🎤 Search Song on Spotify"},
    'tab2_title': {'TR': "🎯 Proje Detayları", 'EN': "🎯 Project Details"},

    'tab1_header': {'TR': "Spotify'da Şarkı Arayarak Popülerlik Tahmin Et", 'EN': "Predict Popularity by Searching on Spotify"},
    'tab1_api_warning': {'TR': "Uygulamanın bu özelliği kullanabilmesi için Spotify API bilgilerinin ayarlanmış olması gerekmektedir.", 'EN': "Spotify API credentials must be configured to use this feature."},
    'search_form_label': {'TR': "Şarkı Adı ve/veya Sanatçı", 'EN': "Song Name and/or Artist"},
    'search_button': {'TR': 'Ara', 'EN': 'Search'},
    'search_results_header': {'TR': "Arama Sonuçları", 'EN': "Search Results"},
    'artist_label': {'TR': "Sanatçı", 'EN': "Artist"},
    'predict_button': {'TR': "Popülerliği Tahmin Et", 'EN': "Predict Popularity"},
    'prediction_header': {'TR': "🔮 '{track_name}' için Tahmin Sonucu", 'EN': "🔮 Prediction Result for '{track_name}'"},
    'metric_prediction': {'TR': "Modelin Tahmini Popülerlik Puanı", 'EN': "Model's Predicted Popularity Score"},
    'metric_real': {'TR': "Gerçek Spotify Popülerlik Puanı", 'EN': "Actual Spotify Popularity Score"},
    'feedback_hit': {'TR': "Bu şarkı bir hit potansiyeli taşıyor! 🚀", 'EN': "This song has hit potential! 🚀"},
    'feedback_popular': {'TR': "Bu şarkı oldukça popüler olabilir. 👍", 'EN': "This song could be quite popular. 👍"},
    'feedback_niche': {'TR': "Bu şarkı daha niş bir kitleye hitap edebilir. 🎵", 'EN': "This song might appeal to a more niche audience. 🎵"},
    'audio_features_error': {'TR': "Bu şarkının ses özellikleri alınamadı.", 'EN': "Could not retrieve audio features for this song."},
    'api_error_403': {'TR': "API Hatası (403 Forbidden): Spotify isteği reddetti. Lütfen Streamlit Secrets'a eklediğiniz bilgileri kontrol edin.", 'EN': "API Error (403 Forbidden): Spotify rejected the request. Please check the credentials you added to Streamlit Secrets."},
    'api_error_generic': {'TR': "Bir hata oluştu. API bağlantısı kurulamadı. Hata: {e}", 'EN': "An error occurred. Could not connect to the API. Error: {e}"},
    'close_button': {'TR': 'Kapat', 'EN': 'Close'},

    'tab2_header': {'TR': "Projenin Amacı ve Teknik Detaylar", 'EN': "Project Goal and Technical Details"},
    'tab2_text': {'TR': """Bu projenin amacı, bir şarkının Spotify'daki popülerliğini, Spotify API tarafından sağlanan ses özelliklerine göre tahmin etmektir. \n- **Model:** `XGBoost Regressor`\n- **Veri Seti:** Spotify Features (Kaggle) & Live Spotify API""", 'EN': """The goal of this project is to predict the popularity of a song on Spotify based on its audio features provided by the Spotify API. \n- **Model:** `XGBoost Regressor`\n- **Dataset:** Spotify Features (Kaggle) & Live Spotify API"""},
    'expander_title': {'TR': "🎵 Ses Özellikleri Ne Anlama Geliyor?", 'EN': "🎵 What Do the Audio Features Mean?"},
    'expander_text': {'TR': """- **Dans Edilebilirlik (Danceability):** Dans etmeye ne kadar uygun olduğunu açıklar.\n- **Enerji (Energy):** Yoğunluk ve aktivitenin algısal bir ölçüsüdür.\n- **Gürültü (Loudness):** Genel ses yüksekliği (desibel - dB).\n- **Akustiklik (Acousticness):** Parçanın akustik olup olmadığının bir ölçüsü.\n- **Enstrümantallik (Instrumentalness):** Parçanın vokal içerip içermediğini tahmin eder.\n- **Pozitiflik (Valence):** Müziksel pozitifliği (mutlu, neşeli) açıklar.\n- **Tempo:** Dakikadaki vuruş sayısı (BPM).""", 'EN': """- **Danceability:** Describes how suitable a track is for dancing.\n- **Energy:** A perceptual measure of intensity and activity.\n- **Loudness:** The overall loudness of a track in decibels (dB).\n- **Acousticness:** A measure of whether the track is acoustic.\n- **Instrumentalness:** Predicts whether a track contains no vocals.\n- **Valence:** Describes the musical positiveness (e.g., happy, cheerful) conveyed by a track.\n- **Tempo:** The overall estimated tempo of a track in beats per minute (BPM)."""}
}

if 'tracks' not in st.session_state:
    st.session_state.tracks = []
if 'selected_track' not in st.session_state:
    st.session_state.selected_track = None
if 'lang' not in st.session_state:
    st.session_state.lang = 'TR'

st.sidebar.selectbox(
    label=texts['language_label'][st.session_state.lang],
    options=['TR', 'EN'],
    key='lang'
)
lang = st.session_state.lang

st.set_page_config(
    page_title=texts['page_title'][lang],
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_resources():
    try:
        model = joblib.load('model.pkl')
        model_features = joblib.load('model_features.pkl')
        return model, model_features
    except FileNotFoundError:
        return None, None

model, model_features = load_resources()

st.title(texts['main_title'][lang])

if not model or not model_features:
    st.error(texts['model_error'][lang])
    st.stop()

try:
    client_id = st.secrets["SPOTIPY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]
    st.sidebar.success(texts['sidebar_api_success'][lang])
except (KeyError, FileNotFoundError):
    st.sidebar.error(texts['sidebar_api_error'][lang])
    st.sidebar.info(texts['sidebar_api_info'][lang])
    client_id = ""
    client_secret = ""

tab1, tab2 = st.tabs([
    f"🎤 **{texts['tab1_title'][lang]}**", 
    f"🎯 **{texts['tab2_title'][lang]}**"
])

with tab1:
    st.header(texts['tab1_header'][lang])
    
    if not client_id or not client_secret:
        st.toast(texts['tab1_api_warning'][lang], icon='⚠️')
    else:
        try:
            client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

            with st.form(key='search_form'):
                search_query = st.text_input(texts['search_form_label'][lang])
                search_button = st.form_submit_button(label=texts['search_button'][lang])

            if search_button and search_query:
                results = sp.search(q=search_query, type='track', limit=10)
                st.session_state.tracks = results['tracks']['items']
                st.session_state.selected_track = None
            
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

            if st.session_state.selected_track:
                track = st.session_state.selected_track
                
                with st.dialog(texts['prediction_header'][lang].format(track_name=track['name'])):
                    audio_features = sp.audio_features(track['id'])[0]
                    
                    if audio_features:
                        input_dict = {k: audio_features.get(k) for k in model_features}
                        input_df = pd.DataFrame([input_dict])[model_features]
                        prediction = model.predict(input_df)
                        popularity_score = int(prediction[0])

                        col_pred, col_real = st.columns(2)
                        col_pred.metric(label=texts['metric_prediction'][lang], value=popularity_score)
                        col_pred.progress(popularity_score)
                        
                        col_real.metric(label=texts['metric_real'][lang], value=track['popularity'])
                        col_real.progress(track['popularity'])

                        if popularity_score > 80:
                            st.success(texts['feedback_hit'][lang])
                        elif popularity_score > 60:
                            st.info(texts['feedback_popular'][lang])
                        else:
                            st.warning(texts['feedback_niche'][lang])
                    else:
                        st.error(texts['audio_features_error'][lang])

                    if st.button(texts['close_button'][lang]):
                        st.session_state.selected_track = None
                        st.rerun()

        except Exception as e:
            error_message = str(e)
            if "403" in error_message or "Forbidden" in error_message:
                st.toast(texts['api_error_403'][lang], icon='🚨')
            else:
                st.toast(texts['api_error_generic'][lang].format(e=e), icon='🚨')

with tab2:
    st.header(texts['tab2_header'][lang])
    st.write(texts['tab2_text'][lang])
    with st.expander(texts['expander_title'][lang]):
        st.markdown(texts['expander_text'][lang])
