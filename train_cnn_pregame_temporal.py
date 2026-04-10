# train_cnn_pregame_chronological.py
# Compare a 1D CNN against the existing SVM baseline for StatLine pre-game NBA prediction
# using a chronological 80/20 split only.

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, Tuple

import argparse
import json
import random

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Reuse your existing StatLine pre-game feature pipeline
from train_svm_momentum_svm import (
    configure_nba_session,
    load_season_team_logs,
    compute_last_season_team_averages,
    add_current_season_momentum,
    blend_last_and_current,
    add_elo_column,
)

# TensorFlow / Keras for the 1D CNN
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


# ----------------------------
# Reproducibility
# ----------------------------

def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------
# Build game-level examples with dates
# ----------------------------

def build_game_level_examples_with_dates(
    df_blend: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build game-level examples and keep metadata like game_id and game_date
    so we can do chronological splitting.

    Returns:
        X: features
        y: labels
        meta: DataFrame with game_id and game_date aligned to X/y
    """
    print("Building game-level training examples with dates...")
    df = df_blend.sort_values(["GAME_DATE", "GAME_ID"]).copy()

    feat_cols = [
        "BLEND_FG_PCT",
        "BLEND_FG3_PCT",
        "BLEND_FT_PCT",
        "BLEND_PTS",
        "BLEND_REB",
        "BLEND_AST",
        "BLEND_TOV",
        "BLEND_PLUS_MINUS",
        "BLEND_NET_MARGIN",
        "RECENT_WIN_PCT",
        "ELO_PRE",
    ]

    games = []
    labels = []
    metas = []

    for game_id, g in df.groupby("GAME_ID"):
        if len(g) != 2:
            continue

        home_rows = g[g["MATCHUP"].str.contains(" vs. ", na=False)]
        away_rows = g[g["MATCHUP"].str.contains(" @ ", na=False)]

        if len(home_rows) != 1 or len(away_rows) != 1:
            continue

        home = home_rows.iloc[0]
        away = away_rows.iloc[0]

        label = 1 if home["WL"] == "W" else 0

        feat_vals = {}
        for col in feat_cols:
            h_val = float(home.get(col, np.nan))
            a_val = float(away.get(col, np.nan))
            feat_vals[f"{col}_DIFF"] = h_val - a_val

        games.append(feat_vals)
        labels.append(label)
        metas.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": pd.to_datetime(home["GAME_DATE"]),
            }
        )

    X_raw = pd.DataFrame(games)
    y_raw = pd.Series(labels, dtype=int)
    meta_raw = pd.DataFrame(metas)

    mask = ~X_raw.isna().any(axis=1)
    X = X_raw[mask].reset_index(drop=True)
    y = y_raw[mask].reset_index(drop=True)
    meta = meta_raw[mask].reset_index(drop=True)

    print(f"Built {len(X)} game-level examples (after dropping NaN rows).")
    return X, y, meta


def build_pregame_dataset_with_dates(
    last_season_label: str,
    current_season_label: str,
    w_last: float = 0.15,
    w_cur: float = 0.85,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build the exact same pre-game feature matrix used by the current SVM pipeline,
    while keeping dates for chronological evaluation.
    """
    configure_nba_session()

    print(f"Loading last season: {last_season_label}")
    df_last = load_season_team_logs(last_season_label)

    print(f"Loading current season: {current_season_label}")
    df_cur = load_season_team_logs(current_season_label)

    print("Combining seasons and adding Elo...")
    df_all = pd.concat([df_last, df_cur], ignore_index=True)
    df_all = add_elo_column(df_all)

    print("Keeping current season rows with Elo...")
    df_cur_with_elo = df_all[df_all["SEASON"] == current_season_label].copy()

    print("Computing last-season team averages...")
    last_avg = compute_last_season_team_averages(df_last)

    print("Adding current-season momentum...")
    df_cur_mom = add_current_season_momentum(df_cur_with_elo)

    print("Blending last season and current season...")
    df_blend = blend_last_and_current(
        df_cur_mom,
        last_avg,
        w_last=w_last,
        w_cur=w_cur,
        shrink_games=20,
    )

    print("Building game-level examples...")
    X, y, meta = build_game_level_examples_with_dates(df_blend)

    # enforce chronological order
    order = np.argsort(meta["GAME_DATE"].values)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    meta = meta.iloc[order].reset_index(drop=True)

    print(f"Final dataset shape: X={X.shape}, y={y.shape}, meta={meta.shape}")
    return X, y, meta


# ----------------------------
# Chronological split
# ----------------------------

def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    train_frac: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    n = len(X)
    split_idx = int(n * train_frac)

    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_val = X.iloc[split_idx:].reset_index(drop=True)

    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_val = y.iloc[split_idx:].reset_index(drop=True)

    meta_train = meta.iloc[:split_idx].reset_index(drop=True)
    meta_val = meta.iloc[split_idx:].reset_index(drop=True)

    return X_train, X_val, y_train, y_val, meta_train, meta_val


# ----------------------------
# SVM baseline
# ----------------------------

def train_svm_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train the same style of SVM baseline used in your current project.
    """
    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=2.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    print("\nTraining SVM baseline...")
    svm_model.fit(X_train, y_train)
    return svm_model


# ----------------------------
# KNN baseline
# ----------------------------

def train_knn_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = 10,
) -> Pipeline:
    """
    Train a KNN baseline on the same feature set.
    Uses scaling first because KNN is distance-based.
    """
    knn_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights="distance",
                ),
            ),
        ]
    )

    print("\nTraining KNN baseline...")
    knn_model.fit(X_train, y_train)
    return knn_model


# ----------------------------
# CNN model
# ----------------------------

def build_cnn_model(input_length: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    """
    Build a small 1D CNN for short tabular sequences.
    Input shape: (num_features, 1)
    """
    model = Sequential(
        [
            Input(shape=(input_length, 1)),
            Conv1D(filters=32, kernel_size=3, activation="relu", padding="same"),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.30),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def prepare_cnn_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize features, then reshape for Conv1D:
    (n_samples, n_features) -> (n_samples, n_features, 1)
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

    return X_train_cnn, X_val_cnn, scaler


def train_cnn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> Tuple[tf.keras.Model, StandardScaler, tf.keras.callbacks.History]:
    """
    Train the 1D CNN on the same features used by the SVM.
    """
    print("\nPreparing CNN inputs...")
    X_train_cnn, X_val_cnn, cnn_scaler = prepare_cnn_inputs(X_train, X_val)

    print("Building CNN model...")
    cnn_model = build_cnn_model(
        input_length=X_train.shape[1],
        learning_rate=learning_rate,
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    )

    print("\nTraining 1D CNN...")
    history = cnn_model.fit(
        X_train_cnn,
        y_train.to_numpy(),
        validation_data=(X_val_cnn, y_val.to_numpy()),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    return cnn_model, cnn_scaler, history


# ----------------------------
# Evaluation helpers
# ----------------------------

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_svm(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, object]:
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_binary_metrics(y_val.to_numpy(), y_pred)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_val, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }


def evaluate_knn(
    model: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, object]:
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    metrics = compute_binary_metrics(y_val.to_numpy(), y_pred)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_val, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }


def evaluate_cnn(
    model: tf.keras.Model,
    scaler: StandardScaler,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, object]:
    X_val_scaled = scaler.transform(X_val)
    X_val_cnn = X_val_scaled.reshape(X_val_scaled.shape[0], X_val_scaled.shape[1], 1)

    y_prob = model.predict(X_val_cnn, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_binary_metrics(y_val.to_numpy(), y_pred)

    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_val, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist(),
    }


# ----------------------------
# Saving
# ----------------------------

def save_svm_bundle(
    model: Pipeline,
    feature_names: list[str],
    out_path: str,
    metadata: Dict[str, object],
) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "metadata": metadata,
    }
    joblib.dump(bundle, out_file)
    print(f"SVM bundle saved to: {out_file.resolve()}")


def save_cnn_artifacts(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_names: list[str],
    model_path: str,
    scaler_path: str,
    metadata_path: str,
    metadata: Dict[str, object],
) -> None:
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)
    metadata_file = Path(metadata_path)

    model_file.parent.mkdir(parents=True, exist_ok=True)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_file)
    joblib.dump(scaler, scaler_file)

    payload = {
        "feature_names": feature_names,
        "metadata": metadata,
    }
    metadata_file.write_text(json.dumps(payload, indent=2))

    print(f"CNN model saved to: {model_file.resolve()}")
    print(f"CNN scaler saved to: {scaler_file.resolve()}")
    print(f"CNN metadata saved to: {metadata_file.resolve()}")


def save_comparison_csv(
    svm_metrics: Dict[str, float],
    cnn_metrics: Dict[str, float],
    knn_metrics: Dict[str, float],
    out_csv: str,
) -> None:
    rows = [
        {"model": "SVM", **svm_metrics},
        {"model": "CNN_1D", **cnn_metrics},
        {"model": "KNN", **knn_metrics},
    ]
    df = pd.DataFrame(rows)
    out_file = Path(out_csv)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Comparison CSV saved to: {out_file.resolve()}")


# ----------------------------
# Main chronological comparison runner
# ----------------------------

def run_comparison(
    last_season: str = "2024-25",
    current_season: str = "2025-26",
    random_state: int = 42,
    w_last: float = 0.15,
    w_cur: float = 0.85,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
) -> None:
    set_global_seed(random_state)

    print("Building shared pre-game dataset...")
    X, y, meta = build_pregame_dataset_with_dates(
        last_season_label=last_season,
        current_season_label=current_season,
        w_last=w_last,
        w_cur=w_cur,
    )

    feature_names = list(X.columns)

    X_train, X_val, y_train, y_val, meta_train, meta_val = chronological_split(
        X, y, meta, train_frac=0.8
    )

    print("\nUsing chronological 80/20 split:")
    print(f"Train date range: {meta_train['GAME_DATE'].min()} -> {meta_train['GAME_DATE'].max()}")
    print(f"Val date range:   {meta_val['GAME_DATE'].min()} -> {meta_val['GAME_DATE'].max()}")
    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val shape:   X={X_val.shape}, y={y_val.shape}")

    # ---- Train SVM ----
    svm_model = train_svm_baseline(X_train, y_train)
    svm_results = evaluate_svm(svm_model, X_val, y_val)

    print("\n===== SVM RESULTS =====")
    print(svm_results["report"])
    print("Confusion Matrix:", svm_results["confusion_matrix"])
    print("Metrics:", svm_results["metrics"])

    # ---- Train CNN ----
    cnn_model, cnn_scaler, _ = train_cnn_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    cnn_results = evaluate_cnn(cnn_model, cnn_scaler, X_val, y_val)

    print("\n===== CNN RESULTS =====")
    print(cnn_results["report"])
    print("Confusion Matrix:", cnn_results["confusion_matrix"])
    print("Metrics:", cnn_results["metrics"])

    # ---- Train KNN ----
    knn_model = train_knn_baseline(X_train, y_train, n_neighbors=10)
    knn_results = evaluate_knn(knn_model, X_val, y_val)

    print("\n===== KNN RESULTS =====")
    print(knn_results["report"])
    print("Confusion Matrix:", knn_results["confusion_matrix"])
    print("Metrics:", knn_results["metrics"])

    # ---- Print final comparison ----
    print("\n===== FINAL MODEL COMPARISON =====")
    comparison_df = pd.DataFrame(
        [
            {"model": "SVM", **svm_results["metrics"]},
            {"model": "CNN_1D", **cnn_results["metrics"]},
            {"model": "KNN", **knn_results["metrics"]},
        ]
    )
    print(comparison_df.to_string(index=False))

    # ---- Save artifacts ----
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S")

    models_dir = Path("models")
    results_dir = Path("outputs")
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    metadata = {
        "last_season": last_season,
        "current_season": current_season,
        "weights": {"last": w_last, "current": w_cur},
        "evaluation": {
            "type": "chronological_80_20",
            "train_start": str(meta_train["GAME_DATE"].min()),
            "train_end": str(meta_train["GAME_DATE"].max()),
            "val_start": str(meta_val["GAME_DATE"].min()),
            "val_end": str(meta_val["GAME_DATE"].max()),
        },
        "random_state": random_state,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "trained_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }

    save_svm_bundle(
        model=svm_model,
        feature_names=feature_names,
        out_path=str(models_dir / f"svm_baseline_chronological_{timestamp}.pkl"),
        metadata=metadata,
    )

    save_cnn_artifacts(
        model=cnn_model,
        scaler=cnn_scaler,
        feature_names=feature_names,
        model_path=str(models_dir / f"cnn_pregame_chronological_{timestamp}.keras"),
        scaler_path=str(models_dir / f"cnn_pregame_chronological_scaler_{timestamp}.pkl"),
        metadata_path=str(models_dir / f"cnn_pregame_chronological_metadata_{timestamp}.json"),
        metadata=metadata,
    )

    save_comparison_csv(
        svm_metrics=svm_results["metrics"],
        cnn_metrics=cnn_results["metrics"],
        knn_metrics=knn_results["metrics"],
        out_csv=str(results_dir / f"pregame_model_comparison_chronological_{timestamp}.csv"),
    )

    summary_payload = {
        "svm": {
            "metrics": svm_results["metrics"],
            "confusion_matrix": svm_results["confusion_matrix"],
        },
        "cnn": {
            "metrics": cnn_results["metrics"],
            "confusion_matrix": cnn_results["confusion_matrix"],
        },
        "knn": {
            "metrics": knn_results["metrics"],
            "confusion_matrix": knn_results["confusion_matrix"],
        },
        "metadata": metadata,
    }
    summary_json = results_dir / f"pregame_model_comparison_chronological_{timestamp}.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2))
    print(f"Detailed comparison JSON saved to: {summary_json.resolve()}")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and compare StatLine pre-game SVM baseline vs 1D CNN using a chronological split."
    )
    parser.add_argument("--last_season", type=str, default="2024-25")
    parser.add_argument("--current_season", type=str, default="2025-26")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--w_last", type=float, default=0.15)
    parser.add_argument("--w_cur", type=float, default=0.85)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)

    args = parser.parse_args()

    run_comparison(
        last_season=args.last_season,
        current_season=args.current_season,
        random_state=args.random_state,
        w_last=args.w_last,
        w_cur=args.w_cur,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )