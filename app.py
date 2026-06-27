
import os
import zipfile
import tempfile
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import fingerprint as fp

DB_PATH = "fingerprint_db.pkl"
SONGS_FOLDER = "songs"

st.set_page_config(page_title="Audio Fingerprint Identifier", layout="wide")
st.title("🎵 Audio Fingerprint Identifier")
st.caption("Shazam-style song recognition — EE200 Q3B")


@st.cache_resource
def setup():
    try:
        if os.path.exists(DB_PATH):
            db, songs = fp.load_database(DB_PATH)
            return db, songs
        elif os.path.isdir(SONGS_FOLDER):
            with st.spinner("Building database from songs folder..."):
                db, songs = fp.build_database(SONGS_FOLDER, DB_PATH)
            return db, songs
        else:
            return {}, []
    except Exception as e:
        st.error(f"Setup error: {e}")
        return {}, []


db, songs = setup()

with st.sidebar:
    st.header("Database")
    st.write(f"**{len(songs)} songs indexed**")
    if songs:
        with st.expander("Show song list"):
            for s in songs:
                st.write("•", s)

mode = st.tabs(["🎧 Single-clip mode", "📦 Batch mode"])


def plot_spectrogram(res):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.pcolormesh(res["t"], res["f"], res["Sxx_db"],
                       shading="gouraud", cmap="magma")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram"); fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    return fig


def plot_constellation(res):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("black")
    ax.scatter(res["t"][res["time_idx"]], res["f"][res["freq_idx"]],
               s=8, c="cyan")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Constellation ({len(res['freq_idx'])} peaks)")
    plt.tight_layout()
    return fig


def plot_histogram(offsets, best_song):
    fig, ax = plt.subplots(figsize=(8, 3))
    if best_song and best_song in offsets:
        hist = offsets[best_song]
        ax.bar(list(hist.keys()), list(hist.values()), color="tomato")
        ax.set_title(f"Offset histogram — {best_song}")
    ax.set_xlabel("Offset (frames)"); ax.set_ylabel("Hash count")
    plt.tight_layout()
    return fig


with mode[0]:
    st.subheader("Identify a single query clip")
    if not songs:
        st.warning("No songs indexed yet.")
    else:
        uploaded = st.file_uploader("Upload a query clip",
                                     type=["wav","mp3","flac","ogg"],
                                     key="single")
        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            st.audio(uploaded)
            try:
                with st.spinner("Matching..."):
                    y = fp.load_audio(tmp_path)
                    best_song, best_score, offsets, res = fp.match_query(y, db)
                if best_song and best_score >= 3:
                    st.success(f"✅ Matched: **{best_song}** (score = {best_score})")
                else:
                    st.error(f"❌ No confident match (best score = {best_score})")
                col1, col2 = st.columns(2)
                with col1: st.pyplot(plot_spectrogram(res))
                with col2: st.pyplot(plot_constellation(res))
                st.pyplot(plot_histogram(offsets, best_song))
            except Exception as e:
                st.error(f"Matching error: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


with mode[1]:
    st.subheader("Identify many clips at once")
    if not songs:
        st.warning("No songs indexed yet.")
    else:
        batch_files = st.file_uploader("Upload clips or a .zip",
                                        type=["wav","mp3","flac","ogg","zip"],
                                        accept_multiple_files=True,
                                        key="batch")
        if batch_files and st.button("Run batch identification"):
            rows = []
            tmp_dir = tempfile.mkdtemp()
            clip_paths = []
            for f_ in batch_files:
                if f_.name.lower().endswith(".zip"):
                    zpath = os.path.join(tmp_dir, f_.name)
                    with open(zpath, "wb") as out: out.write(f_.read())
                    with zipfile.ZipFile(zpath) as zf: zf.extractall(tmp_dir)
                    for root, _, files in os.walk(tmp_dir):
                        for fn in files:
                            if fn.lower().endswith((".wav",".mp3",".flac",".ogg")):
                                clip_paths.append(os.path.join(root, fn))
                else:
                    p = os.path.join(tmp_dir, f_.name)
                    with open(p, "wb") as out: out.write(f_.read())
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
                progress.progress((i + 1) / len(clip_paths))
            df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.dataframe(df)
            st.download_button("⬇️ Download results.csv",
                                df.to_csv(index=False).encode(),
                                "results.csv", "text/csv")
