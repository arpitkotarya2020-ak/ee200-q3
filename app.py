"""
app.py
Streamlit front-end for the audio fingerprinting identifier (Q3B).

Two modes (as required by the project spec):
  1. Single-clip mode: upload one query clip, see spectrogram, constellation,
     offset histogram, and the matched song name.
  2. Batch mode: upload several query clips, get results.csv with
     columns [filename, prediction].

Run locally with:   streamlit run app.py
Deploy on Streamlit Community Cloud by pointing it at this file in your repo.
"""

import os
import io
import zipfile
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import fingerprint as fp

DB_PATH = "fingerprint_db.pkl"
SONGS_FOLDER = "songs"   # ship the indexed song library alongside the app

st.set_page_config(page_title="Audio Fingerprint Identifier", layout="wide")
st.title("🎵 Audio Fingerprint Identifier")
st.caption("Shazam-style song recognition using spectrogram constellation maps "
           "and paired-peak hashing (EE200 Q3).")


# ---------------------------------------------------------------------
# Load or build the database once, cache it
# ---------------------------------------------------------------------

@st.cache_resource
def get_database():
    if os.path.exists(DB_PATH):
        db, songs = fp.load_database(DB_PATH)
    elif os.path.isdir(SONGS_FOLDER):
        with st.spinner("Indexing song library for the first time..."):
            db, songs = fp.build_database(SONGS_FOLDER, DB_PATH)
    else:
        db, songs = {}, []
    return db, songs


db, songs = get_database()

with st.sidebar:
    st.header("Database")
    st.write(f"**{len(songs)} songs indexed**")
    if songs:
        with st.expander("Show song list"):
            for s in songs:
                st.write("•", s)
    st.divider()
    st.write("To re-index (e.g. after adding songs to the `songs/` folder), "
             "delete `fingerprint_db.pkl` and restart the app.")

mode = st.tabs(["🎧 Single-clip mode", "📦 Batch mode"])


# ---------------------------------------------------------------------
# Helper: plot spectrogram + constellation overlay
# ---------------------------------------------------------------------

def plot_spectrogram_with_peaks(result):
    f, t, Sxx_db = result["f"], result["t"], result["Sxx_db"]
    fi, ti = result["freq_idx"], result["time_idx"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    return fig


def plot_constellation(result):
    f, t = result["f"], result["t"]
    fi, ti = result["freq_idx"], result["time_idx"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("black")
    ax.scatter(t[ti], f[fi], s=8, c="cyan")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation map ({len(fi)} peaks)")
    return fig


def plot_offset_histogram(offsets, best_song):
    fig, ax = plt.subplots(figsize=(8, 3))
    if best_song is not None and best_song in offsets:
        hist = offsets[best_song]
        ax.bar(list(hist.keys()), list(hist.values()), color="tomato")
        ax.set_title(f"Offset histogram — best match: {best_song}")
    else:
        ax.set_title("No match found")
    ax.set_xlabel("Offset (STFT frames)")
    ax.set_ylabel("Matching hash count")
    return fig


# ---------------------------------------------------------------------
# MODE 1: Single-clip
# ---------------------------------------------------------------------

with mode[0]:
    st.subheader("Identify a single query clip")

    if not songs:
        st.warning("No songs indexed yet. Add audio files to a `songs/` folder "
                    "next to this app (and ship it with the deployment) so it can "
                    "build the database on first run.")
    else:
        uploaded = st.file_uploader(
            "Upload a query clip (wav / mp3 / flac / ogg)",
            type=["wav", "mp3", "flac", "ogg"],
            key="single",
        )

        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            st.audio(uploaded)

            with st.spinner("Fingerprinting and matching..."):
                y = fp.load_audio(tmp_path)
                best_song, best_score, offsets, result = fp.match_query(y, db)

            if best_song is not None and best_score >= 3:
                st.success(f"✅ Matched: **{best_song}**  (score = {best_score} aligned hashes)")
            else:
                st.error(f"❌ No confident match found (best score = {best_score})")

            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(plot_spectrogram_with_peaks(result))
            with col2:
                st.pyplot(plot_constellation(result))

            st.pyplot(plot_offset_histogram(offsets, best_song))

            os.remove(tmp_path)


# ---------------------------------------------------------------------
# MODE 2: Batch
# ---------------------------------------------------------------------

with mode[1]:
    st.subheader("Identify many query clips at once")

    if not songs:
        st.warning("No songs indexed yet — see the Single-clip tab for setup notes.")
    else:
        batch_files = st.file_uploader(
            "Upload multiple query clips, or a .zip of clips",
            type=["wav", "mp3", "flac", "ogg", "zip"],
            accept_multiple_files=True,
            key="batch",
        )

        if batch_files:
            run = st.button("Run batch identification")
            if run:
                rows = []
                progress = st.progress(0.0)

                # Expand any uploaded zip files into individual clips
                clip_paths = []
                tmp_dir = tempfile.mkdtemp()
                for f_ in batch_files:
                    if f_.name.lower().endswith(".zip"):
                        zpath = os.path.join(tmp_dir, f_.name)
                        with open(zpath, "wb") as out:
                            out.write(f_.read())
                        with zipfile.ZipFile(zpath) as zf:
                            zf.extractall(tmp_dir)
                        for root, _, files in os.walk(tmp_dir):
                            for fn in files:
                                if fn.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                                    clip_paths.append(os.path.join(root, fn))
                    else:
                        p = os.path.join(tmp_dir, f_.name)
                        with open(p, "wb") as out:
                            out.write(f_.read())
                        clip_paths.append(p)

                for i, path in enumerate(clip_paths):
                    fname = os.path.basename(path)
                    try:
                        y = fp.load_audio(path)
                        best_song, best_score, _, _ = fp.match_query(y, db)
                        prediction = best_song if (best_song is not None and best_score >= 3) else ""
                    except Exception as e:
                        prediction = ""
                    rows.append({"filename": fname, "prediction": prediction})
                    progress.progress((i + 1) / len(clip_paths))

                results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
                st.success(f"Done — {len(results_df)} clips processed.")
                st.dataframe(results_df)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download results.csv",
                    data=csv_bytes,
                    file_name="results.csv",
                    mime="text/csv",
                )
