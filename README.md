🎵 Day 7: Song Popularity Prediction

This is the seventh project of my #30DaysOfAI challenge. The goal is to build a regression model that predicts a song's popularity score on Spotify based on its audio features and present it via an interactive Streamlit web application.
✨ Key Concepts

    Regression with XGBoost: Applying a powerful gradient boosting algorithm to predict a continuous value (popularity score).

    Interactive Machine Learning: Creating a user-friendly web app with Streamlit that allows anyone to interact with the trained model.

    Feature Engineering & Interpretation: Understanding and using specific domain features (audio metrics like danceability, energy, valence) to make predictions.

    Model Deployment: Saving and loading a trained model (.pkl) for use in a separate application, which is a fundamental MLOps concept.

💻 Tech Stack

    Python

    Pandas for data manipulation

    Scikit-learn & XGBoost for modeling

    Streamlit for the web application

    Joblib for model serialization

🚀 How to Run

    Clone the repository:

    git clone https://github.com/SuleymanToklu/Day7-Song-Popularity-Prediction
    cd Day7-Song-Popularity-Prediction

    Install dependencies:

    pip install -r requirements.txt

    Download the dataset:
    Download SpotifyFeatures.csv from this Kaggle page and place it in the project's root directory.

    Train the model:
    Run the training script to generate model.pkl and model_features.pkl.

    python train_model.py

    Run the Streamlit app:

    streamlit run app.py
