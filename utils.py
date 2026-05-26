from os import listdir
from os.path import isfile, join
import scipy
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns


def sliding_window_pd(
        df,
        ws=500,
        overlap=250,
        w_type="hann",
        w_center=True,
        print_stats=False
) -> list:
    """Applies the sliding window algorithm to the DataFrame rows."""
    counter = 0
    windows_list = list()
    for window in df.rolling(window=ws, step=overlap, min_periods=ws,
                             win_type=w_type, center=w_center):
        if window[window.columns[0]].count() >= ws:
            if print_stats:
                print("Print Window:", counter)
                print("Number of samples:", window[window.columns[0]].count())
            windows_list.append(window)
        counter += 1
    if print_stats:
        print("List number of window instances:", len(windows_list))
    return windows_list


def apply_filter(
        arr,
        order=5,
        wn=0.1,
        filter_type="lowpass"
) -> np.ndarray:
    """Applies Butterworth filter to a signal array."""
    fbd_filter = scipy.signal.butter(N=order, Wn=wn, btype=filter_type, output="sos")
    filtered_signal = scipy.signal.sosfiltfilt(sos=fbd_filter, x=arr, padlen=0)
    return filtered_signal


def filter_instances(instances_list, order, wn, filter_type) -> list:
    """Applies filter to a list of window DataFrames."""
    filtered_instances_list = list()
    for item in instances_list:
        filtered_instance = item.apply(apply_filter, args=(order, wn, filter_type))
        filtered_instances_list.append(filtered_instance)
    print("Number of filtered instances in the list:", len(filtered_instances_list))
    return filtered_instances_list


def flatten_instances_df(instances_list: list) -> pd.DataFrame:
    """Flattens each window instance into a row and returns a DataFrame."""
    flattened_instances_list = list()
    for item in instances_list:
        instance = item.to_numpy().flatten()
        flattened_instances_list.append(instance)
    df = pd.DataFrame(flattened_instances_list)
    return df


def df_rebase(df: pd.DataFrame, target_list: list, ref_list: list) -> pd.DataFrame:
    """Reorders and renames DataFrame columns."""
    print("Initial columns:", list(df.columns))
    if are_lists_equal(list(df.columns), ref_list):
        pass
    else:
        if len(target_list) == len(ref_list):
            df = df[target_list]
            rename_dict = dict(zip(target_list, ref_list))
            df = df.rename(columns=rename_dict)
        else:
            print("The length of the target list and the reference list is not equal.")
    print("Processed columns:", list(df.columns))
    return df


def rename_df_column_values(np_array: np.ndarray, y: list, columns_names: tuple = ("acc_x", "acc_y", "acc_z")):
    """Creates a DataFrame with encoded y labels."""
    arr_y = np.array(y)
    unique_values_list = np.unique(arr_y)
    df = pd.DataFrame(np_array, columns=columns_names)
    df["y"] = y
    for idx, x in enumerate(unique_values_list):
        df["y"] = np.where(df["y"] == x, idx, df["y"])
    return df


def are_lists_equal(list1: list, list2: list) -> bool:
    return set(list1) == set(list2)


def encode_labels(instances_list) -> np.ndarray:
    """Encodes target labels with LabelEncoder."""
    le = preprocessing.LabelEncoder()
    le.fit(instances_list)
    instances_arr = le.transform(instances_list)
    return instances_arr


def list_files_in_folder(folder_path) -> list:
    """Returns list of CSV files in a folder."""
    files_list = list()
    for f in listdir(folder_path):
        if isfile(join(folder_path, f)):
            if f.endswith(".csv"):
                files_list.append(f)
    return files_list


# ============================================================================
# Project additions (used by the analysis notebooks)
# ============================================================================

import os
import contextlib
import io
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)


def silent_rebase(df, target, ref):
    """Calls `df_rebase` but suppresses its `Initial/Processed columns` debug prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return df_rebase(df, target, ref)


def load_segments(coll, imu_location, sensor="AccGyr", split="Protocol"):
    """Fetches segments for one IMU and returns (df, label, subject) triplets.

    Args:
        coll: pymongo collection.
        imu_location: "hand", "chest" or "ankle".
        sensor: sensor configuration stored in the docs.
        split: dataset split ("Protocol").

    Returns:
        List of dicts {"df": DataFrame, "label": str, "subject": str}.
    """
    query = {"imu_location": imu_location, "sensor": sensor, "split": split}
    segments = []
    for d in coll.find(query):
        segments.append({"df": pd.DataFrame(d["data"]),
                         "label": d["activity_label"],
                         "subject": d["subject"]})
    return segments


def window_segments(segments, ws, overlap, w_type):
    """Slices each segment into fixed-size windows, keeping the labels parallel.

    Args:
        segments: List of segment dicts with ``df`` (DataFrame), ``label``
            (str), and ``subject`` (str) keys.
        ws: Window size in samples.
        overlap: Step size in samples (``ws/2`` for 50% overlap).
        w_type: Window function passed to ``sliding_window_pd`` (e.g.
            ``"hann"``).

    Returns:
        tuple: ``(windows, y, subjects)`` where ``windows`` is the list of
        window DataFrames and ``y``/``subjects`` are parallel ``np.ndarray``
        with the activity label and subject id of each window.
    """
    windows, y, subjects = [], [], []
    for s in segments:
        for w in sliding_window_pd(s["df"], ws=ws, overlap=overlap,
                                   w_type=w_type, print_stats=False):
            windows.append(w.reset_index(drop=True))
            y.append(s["label"]); subjects.append(s["subject"])
    return windows, np.array(y), np.array(subjects)


def filter_windows(windows, order, wn, ftype):
    """Applies a Butterworth filter column-wise to every window.

    Args:
        windows: List of window DataFrames, each shaped ``(ws, n_channels)``.
        order: Filter order (passed to ``scipy.signal.butter`` via
            ``utils.apply_filter``).
        wn: Normalised cutoff frequency in ``(0, 1)``, where ``1`` is the
            Nyquist frequency.
        ftype: Butterworth filter type, e.g. ``"lowpass"``.

    Returns:
        list[pd.DataFrame]: Copies of the input windows with every channel
        replaced by its zero-phase Butterworth-filtered counterpart.
    """
    filtered = []
    for window in windows:
        new_window = window.copy()
        for column in window.columns:
            signal = window[column].to_numpy()
            new_window[column] = apply_filter(signal, order=order, wn=wn,
                                              filter_type=ftype)
        filtered.append(new_window)
    return filtered


def load_combo(coll, imu_locations, sensor="AccGyr", split="Protocol"):
    """Loads segments for multiple IMU locations and concatenates their channels.

    Segments are matched across IMU locations by insertion order - MongoDB
    returns documents sorted by ``_id``, and the ingestion notebook inserts the
    (hand, chest, ankle) triplet of every (subject, activity, segment) together.
    """
    per_loc = {loc: list(coll.find({"imu_location": loc, "sensor": sensor,
                                    "split": split})) for loc in imu_locations}
    n = len(per_loc[imu_locations[0]])
    segments = []
    for i in range(n):
        combined = {}
        for loc in imu_locations:
            for ch, vals in per_loc[loc][i]["data"].items():
                combined[f"{loc}_{ch}"] = vals
        first = per_loc[imu_locations[0]][i]
        segments.append({"df": pd.DataFrame(combined),
                         "label": first["activity_label"],
                         "subject": first["subject"]})
    return segments


def axis_features(x, sr=100):
    """Computes time- and frequency-domain features for a single 1-D signal.

    Args:
        x: 1-D ``np.ndarray`` containing the signal samples of one channel
            of one window.
        sr: Sampling rate in Hz (used to label FFT frequencies).

    Returns:
        dict: Feature name -> scalar value. Includes time-domain statistics
        (mean, std, min, max, median, IQR, RMS, MAD, energy, skewness,
        kurtosis, zero-crossing rate) and frequency-domain features
        (dominant frequency, spectral energy, spectral entropy).
    """
    n = len(x)
    f = {
        "mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x),
        "median": np.median(x),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "rms": np.sqrt(np.mean(x ** 2)),
        "mad": np.mean(np.abs(x - np.mean(x))),
        "energy": np.sum(x ** 2) / n,
        "skew": stats.skew(x), "kurtosis": stats.kurtosis(x),
        "zcr": np.mean(np.abs(np.diff(np.sign(x - np.mean(x)))) > 0),
    }
    spec = np.abs(np.fft.rfft(x - np.mean(x)))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    psd = spec ** 2
    total = psd.sum() + 1e-12
    p = psd / total
    f["dom_freq"] = freqs[np.argmax(psd)]
    f["spec_energy"] = total / n
    f["spec_entropy"] = -np.sum(p * np.log(p + 1e-12))
    return f


def extract_features(window_df, sr=100):
    """Builds a flat feature vector (pandas Series) for one window.

    Concatenates per-axis features (see ``axis_features``) and adds the
    pairwise channel correlations, which act as a coordination cue.

    Args:
        window_df: DataFrame of shape ``(ws, n_channels)`` for one window.
        sr: Sampling rate in Hz.

    Returns:
        pd.Series: Feature vector, indexed by ``"<channel>__<feature>"``
        for axis features and ``"corr__<chan_a>__<chan_b>"`` for the
        upper-triangular pairwise channel correlations.
    """
    feats = {}
    cols = list(window_df.columns)
    for c in cols:
        for k, v in axis_features(window_df[c].to_numpy(), sr=sr).items():
            feats[f"{c}__{k}"] = v
    corr = window_df.corr().to_numpy()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr[i, j]
            feats[f"corr__{cols[i]}__{cols[j]}"] = 0.0 if np.isnan(val) else val
    return pd.Series(feats)


def features_matrix(windows, sr=100):
    """Stacks the per-window feature vectors into a single DataFrame.

    Args:
        windows: List of window DataFrames.
        sr: Sampling rate in Hz, forwarded to ``extract_features``.

    Returns:
        pd.DataFrame: ``(n_windows, n_features)`` feature matrix ready for
        scaling and modelling.
    """
    return pd.DataFrame([extract_features(w, sr=sr) for w in windows])


def train_and_score(model, X_tr, y_tr, X_te, y_te):
    """Trains a model and returns its (accuracy, macro-F1) on the test set."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return accuracy_score(y_te, y_pred), f1_score(y_te, y_pred, average="macro")


def evaluate(model, Xte, yte, title, fname=None, img_dir="img"):
    y_pred = model.predict(Xte)
    acc = accuracy_score(yte, y_pred); f1m = f1_score(yte, y_pred, average="macro")
    print(f"=== {title} ===")
    print(f"accuracy: {acc:.3f} | macro-F1: {f1m:.3f}\n")
    print(classification_report(yte, y_pred, zero_division=0))
    if fname:
        labels = sorted(np.unique(np.concatenate([yte, y_pred])))
        cm = confusion_matrix(yte, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(11, 9))
        ConfusionMatrixDisplay(cm, display_labels=labels).plot(
            ax=ax, xticks_rotation="vertical", cmap="Blues", colorbar=False)
        ax.set_title(title); plt.tight_layout()
        plt.savefig(os.path.join(img_dir, fname), dpi=120); plt.show()
    return acc, f1m


def run_combo_ts(coll, imu_locations, sw_cfg, filt_cfg,
                 train_subjects, test_subjects, rf_cfg):
    """Runs the full time-series pipeline on the concatenated IMU channels."""
    segs = load_combo(coll, imu_locations)
    wins, yy, subj = window_segments(segs, sw_cfg["ws"], sw_cfg["overlap"], sw_cfg["w_type"])
    wins = filter_windows(wins, filt_cfg["order"], filt_cfg["wn"], filt_cfg["type"])
    Xc = flatten_instances_df(wins).to_numpy()
    tr, te = np.isin(subj, train_subjects), np.isin(subj, test_subjects)
    sc = StandardScaler().fit(Xc[tr])
    Xtr, Xte = sc.transform(Xc[tr]), sc.transform(Xc[te])
    model = RandomForestClassifier(n_estimators=rf_cfg["n_estimators"],
                                   max_depth=rf_cfg["max_depth"],
                                   random_state=42, n_jobs=-1).fit(Xtr, yy[tr])
    pred = model.predict(Xte)
    return {"combo": "+".join(imu_locations),
            "n_channels": len(imu_locations) * 6,
            "accuracy": accuracy_score(yy[te], pred),
            "macro_f1": f1_score(yy[te], pred, average="macro")}


def run_combo_fe(coll, imu_locations, sw_cfg, filt_cfg,
                 train_subjects, test_subjects, best_svc):
    """Runs the full feature-engineering pipeline on the concatenated IMU channels."""
    segs = load_combo(coll, imu_locations)
    wins, yy, subj = window_segments(segs, sw_cfg["ws"], sw_cfg["overlap"], sw_cfg["w_type"])
    wins = filter_windows(wins, filt_cfg["order"], filt_cfg["wn"], filt_cfg["type"])
    Xc = features_matrix(wins)
    tr, te = np.isin(subj, train_subjects), np.isin(subj, test_subjects)
    sc = StandardScaler().fit(Xc[tr].values)
    Xtr_, Xte_ = sc.transform(Xc[tr].values), sc.transform(Xc[te].values)
    model = SVC(**best_svc).fit(Xtr_, yy[tr])
    pred = model.predict(Xte_)
    return {"combo": "+".join(imu_locations),
            "n_features": Xc.shape[1],
            "accuracy": accuracy_score(yy[te], pred),
            "macro_f1": f1_score(yy[te], pred, average="macro")}
