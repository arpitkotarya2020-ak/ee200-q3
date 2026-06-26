
import numpy as np
import librosa
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os
import pickle

SR = 22050
NPERSEG = 1024
NOVERLAP = 512
NEIGHBORHOOD = (20, 20)
AMP_THRESHOLD_DB = -40
FAN_OUT = 5
DT_MAX = 200


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr)
    return y


def compute_spectrogram(y, sr=SR):
    f, t, Sxx = spectrogram(y, fs=sr, window='hann',
                             nperseg=NPERSEG, noverlap=NOVERLAP,
                             scaling='spectrum')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, Sxx_db


def find_peaks_2d(Sxx_db, neighborhood=NEIGHBORHOOD, amp_threshold_db=AMP_THRESHOLD_DB):
    max_filt = maximum_filter(Sxx_db, size=neighborhood, mode='constant', cval=-np.inf)
    is_peak = (Sxx_db == max_filt) & (Sxx_db > amp_threshold_db)
    freq_idx, time_idx = np.where(is_peak)
    return freq_idx, time_idx


def generate_hashes(freq_idx, time_idx, fan_out=FAN_OUT, dt_max=DT_MAX, pairs=True):
    order = np.argsort(time_idx)
    freq_idx, time_idx = freq_idx[order], time_idx[order]
    hashes = []
    if not pairs:
        for fidx, tidx in zip(freq_idx, time_idx):
            hashes.append(((int(fidx),), int(tidx)))
        return hashes
    for i in range(len(time_idx)):
        for j in range(1, fan_out + 1):
            if i + j < len(time_idx):
                t1, t2 = time_idx[i], time_idx[i + j]
                dt = t2 - t1
                if 0 < dt <= dt_max:
                    h = (int(freq_idx[i]), int(freq_idx[i + j]), int(dt))
                    hashes.append((h, int(t1)))
    return hashes


def fingerprint_audio(y, sr=SR, pairs=True):
    f, t, Sxx_db = compute_spectrogram(y, sr)
    fi, ti = find_peaks_2d(Sxx_db)
    hashes = generate_hashes(fi, ti, pairs=pairs)
    return {
        "f": f, "t": t, "Sxx_db": Sxx_db,
        "freq_idx": fi, "time_idx": ti,
        "hashes": hashes,
    }


def build_database(song_folder, db_path="fingerprint_db.pkl", pairs=True):
    db = defaultdict(list)
    song_names = []
    for fname in sorted(os.listdir(song_folder)):
        if not fname.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            continue
        name = os.path.splitext(fname)[0]
        song_names.append(name)
        y = load_audio(os.path.join(song_folder, fname))
        fp_data = fingerprint_audio(y, pairs=pairs)
        for h, t1 in fp_data["hashes"]:
            db[h].append((name, t1))
        print(f"  Indexed: {name}")
    with open(db_path, "wb") as fout:
        pickle.dump({"db": dict(db), "songs": song_names}, fout)
    return dict(db), song_names


def load_database(db_path="fingerprint_db.pkl"):
    with open(db_path, "rb") as fin:
        data = pickle.load(fin)
    return data["db"], data["songs"]


def match_query(y, db, sr=SR, pairs=True):
    fp_data = fingerprint_audio(y, sr=sr, pairs=pairs)
    offsets = defaultdict(lambda: defaultdict(int))
    for h, t_query in fp_data["hashes"]:
        if h in db:
            for song_name, t_db in db[h]:
                offset = t_db - t_query
                offsets[song_name][offset] += 1
    best_song, best_score = None, 0
    for song_name, hist in offsets.items():
        peak_count = max(hist.values())
        if peak_count > best_score:
            best_score, best_song = peak_count, song_name
    return best_song, best_score, offsets, fp_data


def match_query_from_path(path, db, sr=SR, pairs=True):
    y = load_audio(path, sr=sr)
    return match_query(y, db, sr=sr, pairs=pairs)
