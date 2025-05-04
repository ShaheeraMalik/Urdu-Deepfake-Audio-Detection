# app.py

import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import zipfile
import os
import pandas as pd
from scipy.special import expit

# ------------------------------------------------------------------
# 1) Caching / loading models + scaler
# ------------------------------------------------------------------
@st.cache_resource
def load_models_and_scaler():
    scaler = joblib.load("scaler.pkl")
    models = {
        "SVM":                 joblib.load("svm_clf.pkl"),
        "Logistic Regression": joblib.load("log_clf.pkl"),
        "Perceptron":          joblib.load("per_clf.pkl"),
        "DNN (2-layer MLP)":    joblib.load("dnn_clf.pkl"),
    }
    return scaler, models

# ------------------------------------------------------------------
# 2) Feature extraction
# ------------------------------------------------------------------
def extract_features_from_path(path, n_mfcc=40):
    """
    Load an audio file and compute 40 MFCCs, returning an 80-dim vector
    of [mean(mfcc), var(mfcc)].
    """
    y, sr = librosa.load(path, sr=None)
    # Use keyword args to avoid positional-argument errors
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.concatenate([mfcc.mean(axis=1), mfcc.var(axis=1)])

# ------------------------------------------------------------------
# 3) Prediction helper
# ------------------------------------------------------------------
def predict(features, scaler, model):
    Xs = scaler.transform([features])
    pred = model.predict(Xs)[0]
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(Xs)[0, pred]
    else:
        # fallback for Perceptron
        score = expit(model.decision_function(Xs)[0])
    return ("Bonafide" if pred == 0 else "Deepfake"), score

# ------------------------------------------------------------------
# 4) Streamlit UI
# ------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Urdu Deepfake Audio Detection", layout="wide")
    st.title("🎙️ Urdu Deepfake Audio Detection")

    scaler, models = load_models_and_scaler()
    model_name = st.sidebar.selectbox("Select model", list(models.keys()))
    model      = models[model_name]

    st.markdown("""
    **Upload either:**
    1. A single audio file (`.wav`, `.mp3`, `.flac`), or  
    2. A ZIP archive of a folder (with subfolders) containing audio files  
    """)

    uploaded = st.file_uploader(
        "📂 Upload audio or ZIP", 
        type=["wav", "mp3", "flac", "zip"], 
        accept_multiple_files=False
    )

    if uploaded is None:
        return

    results = []
    with st.spinner("Processing…"):
        # CASE A: ZIP archive
        if uploaded.name.lower().endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmpdir:
                z = zipfile.ZipFile(uploaded)
                z.extractall(tmpdir)
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        if fn.lower().endswith((".wav", "mp3", "flac")):
                            path = os.path.join(root, fn)
                            feats = extract_features_from_path(path)
                            pred, score = predict(feats, scaler, model)
                            rel = os.path.relpath(path, tmpdir)
                            results.append((rel, pred, score))
        else:
            # CASE B: single file
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp.flush()
                feats = extract_features_from_path(tmp.name)
                pred, score = predict(feats, scaler, model)
                results.append((uploaded.name, pred, score))

    # show audio player if single file
    if uploaded.name.lower().endswith((".wav", "mp3", "flac")):
        st.audio(uploaded)

    # display results
    if results:
        df = pd.DataFrame(results, columns=["Filename", "Prediction", "Confidence"])
        df["Confidence"] = (df["Confidence"] * 100).map("{:.2f}%".format)
        st.subheader("Results")
        st.table(df)

    st.markdown("---")
    st.caption("Built with Streamlit — upload files or a ZIP folder with subfolders.")

if __name__ == "__main__":
    main()
