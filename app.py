
import os
import io
import zipfile
import tempfile
import gdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import fingerprint as fp

DB_PATH = "fingerprint_db.pkl"
SONGS_FOLDER = "songs"

# ← PASTE YOUR GOOGLE DRIVE FOLDER ID HERE
DRIVE_FOLDER_ID = "YOUR_FOLDER_ID_HERE"

st.set_page_config(page_title="Audio Fingerprint Identifier", layout="wide")
st.title("🎵 Audio Fingerprint Identifier")
st.caption("Shazam-style song recognition — EE200 Q3B")


@st.cache_resource
def setup():
    # Download songs from Google Drive if not already present
    if not os.path.isdir(SONGS_FOLDER) or len(os.listdir(SONGS_FOLDER)) == 0:
        os.makedirs(SONGS_FOLDER, exist_ok=True)
        with st.spinner("Downloading song library from Google Drive..."):
            gdown.download_folder(
                id=DRIVE_FOLDER_ID,
                output=SONGS_FOLDER,
                quiet=False,
                use_cookies=False
            )

    # Load or build database
    if os.path.exists(DB_PATH):
        db, songs = fp.load_database(DB_PATH)
    else:
        with st.spinner("Indexing songs..."):
            db, songs = fp.build_database(SONGS_FOLDER, DB_PATH)
    return db, songs


db, songs = setup()

with st.sidebar:
    st.header("Database")
    st.write(f"**{len(songs)} songs indexed**")
    if songs:
        with st.expander("Show song list"):
            for s in songs:
                st.write("•", s)

mode = st.tabs(["🎧 Single-clip mode", "📦 Batch mode"])


def plot_spectrogram(result):
    f, t, Sxx_db = result["f"], result["t"], result["Sxx_db"]
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram"); fig.colorbar(im, ax=ax, label="dB")
    return fig


def plot_constellation(result):
    f, t = result["f"], result["t"]
    fi, ti = result["freq_idx"], result["time_idx"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("black")
    ax.scatter(t[ti], f[fi], s=8, c="cyan")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation ({len(fi)} peaks)")
    return fig


def plot_offset_histogram(offsets, best_song):
    fig, ax = plt.subplots(figsize=(8, 3))
    if best_song and best_song in offsets:
        hist = offsets[best_song]
        ax.bar(list(hist.keys()), list(hist.values()), color="tomato")
        ax.set_title(f"Offset histogram — {best_song}")
    ax.set_xlabel("Offset (frames)"); ax.set_ylabel("Hash count")
    return fig


with mode[0]:
    st.subheader("Identify a single query clip")
    if not songs:
        st.warning("No songs indexed yet.")
    else:
        uploaded = st.file_uploader("Upload a query clip",
                                     type=["wav","mp3","flac","ogg"], key="single")
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False,
                    suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read()); tmp_path = tmp.name
            st.audio(uploaded)
            with st.spinner("Matching..."):
                y = fp.load_audio(tmp_path)
                best_song, best_score, offsets, result = fp.match_query(y, db)
            if best_song and best_score >= 3:
                st.success(f"✅ Matched: **{best_song}** (score = {best_score})")
            else:
                st.error(f"❌ No confident match (best score = {best_score})")
            col1, col2 = st.columns(2)
            with col1: st.pyplot(plot_spectrogram(result))
            with col2: st.pyplot(plot_constellation(result))
            st.pyplot(plot_offset_histogram(offsets, best_song))
            os.remove(tmp_path)


with mode[1]:
    st.subheader("Identify many clips at once")
    if not songs:
        st.warning("No songs indexed yet.")
    else:
        batch_files = st.file_uploader("Upload clips or a .zip",
                                        type=["wav","mp3","flac","ogg","zip"],
                                        accept_multiple_files=True, key="batch")
        if batch_files and st.button("Run batch identification"):
            rows = []
            tmp_dir = tempfile.mkdtemp()
            clip_paths = []
            for f_ in batch_files:
                if f_.name.lower().endswith(".zip"):
                    zpath = os.path.join(tmp_dir, f_.name)
                    with open(zpath,"wb") as out: out.write(f_.read())
                    with zipfile.ZipFile(zpath) as zf: zf.extractall(tmp_dir)
                    for root,_,files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith((".wav",".mp3",".flac",".ogg")):
                                clip_paths.append(os.path.join(root,fn))
                else:
                    p = os.path.join(tmp_dir, f_.name)
                    with open(p,"wb") as out: out.write(f_.read())
                    clip_paths.append(p)
            progress = st.progress(0.0)
            for i, path in enumerate(clip_paths):
                fname = os.path.basename(path)
                try:
                    y = fp.load_audio(path)
                    best_song, best_score, _, _ = fp.match_query(y, db)
                    prediction = best_song if (best_song and best_score >= 3) else ""
                except:
                    prediction = ""
                rows.append({"filename": fname, "prediction": prediction})
                progress.progress((i+1)/len(clip_paths))
            df = pd.DataFrame(rows, columns=["filename","prediction"])
            st.dataframe(df)
            st.download_button("⬇️ Download results.csv",
                                df.to_csv(index=False).encode(),
                                "results.csv", "text/csv")
