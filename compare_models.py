# compare_models.py
#
# Trains and compares three models on NBA game prediction:
#   1. SVM  — RBF kernel (primary baseline)
#   2. KNN  — K-Nearest Neighbors (k tuned via TimeSeriesSplit CV)
#   3. CNN  — 1D Convolutional Neural Network (pure numpy)
#
# Fixes applied vs. earlier version:
#   - KNN k-selection uses TimeSeriesSplit (not StratifiedKFold) so no fold
#     ever trains on future games and tests on past games.
#   - Three-way chronological split: train (70%) / val (15%) / test (15%).
#     CNN checkpoint selection uses only the val set.
#     Final metrics for ALL models are reported on the held-out test set,
#     which is never touched during training or tuning.
#
# Usage:
#   python3 compare_models.py

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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
)

from train_svm_momentum_svm import (
    configure_nba_session,
    load_season_team_logs,
    compute_last_season_team_averages,
    add_current_season_momentum,
    blend_last_and_current,
    add_elo_column,
    build_game_level_examples,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

LAST_SEASON    = "2024-25"
CURRENT_SEASON = "2025-26"
W_LAST         = 0.15
W_CUR          = 0.85
RANDOM_STATE   = 42
OUTPUT_DIR     = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Three-way chronological split fractions
TRAIN_FRAC = 0.70   # first 70%  → train all models
VAL_FRAC   = 0.15   # next  15%  → CNN checkpoint selection ONLY
TEST_FRAC  = 0.15   # last  15%  → final reported metrics (never touched before)


# ===========================================================================
# NUMPY CNN
# ===========================================================================

class Conv1d:
    def __init__(self, in_ch, out_ch, kernel_size, padding=1):
        self.padding = padding
        self.k       = kernel_size
        scale        = np.sqrt(2.0 / (in_ch * kernel_size))
        self.W       = np.random.randn(out_ch, in_ch, kernel_size) * scale
        self.b       = np.zeros(out_ch)
        self.dW = self.db = None

    def forward(self, x):
        self.x_pad = np.pad(x, ((0,0),(0,0),(self.padding, self.padding)))
        B, _, L = x.shape
        L_out = L + 2*self.padding - self.k + 1
        out = np.zeros((B, self.W.shape[0], L_out))
        for i in range(L_out):
            out[:, :, i] = (
                np.einsum('bik,oik->bo', self.x_pad[:, :, i:i+self.k], self.W) + self.b
            )
        return out

    def backward(self, dout):
        B, _, L_out = dout.shape
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dx_pad = np.zeros_like(self.x_pad)
        for i in range(L_out):
            s   = self.x_pad[:, :, i:i+self.k]
            d_i = dout[:, :, i]
            dW      += np.einsum('bo,bik->oik', d_i, s)
            db      += d_i.sum(axis=0)
            dx_pad[:, :, i:i+self.k] += np.einsum('bo,oik->bik', d_i, self.W)
        self.dW = dW / B
        self.db = db / B
        p = self.padding
        return dx_pad[:, :, p:-p] if p > 0 else dx_pad


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    def backward(self, dout):
        return dout * self.mask


class GlobalAvgPool1d:
    def forward(self, x):
        self.L = x.shape[2]
        return x.mean(axis=2)
    def backward(self, dout):
        return np.repeat(dout[:, :, np.newaxis], self.L, axis=2) / self.L


class Linear:
    def __init__(self, in_f, out_f):
        self.W  = np.random.randn(in_f, out_f) * np.sqrt(2.0 / in_f)
        self.b  = np.zeros(out_f)
        self.dW = self.db = None
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, dout):
        B = len(dout)
        self.dW = self.x.T @ dout / B
        self.db = dout.mean(axis=0)
        return dout @ self.W.T


class Dropout:
    def __init__(self, rate=0.3):
        self.rate = rate
        self.training = True
    def forward(self, x):
        if self.training and self.rate > 0:
            self.mask = (np.random.rand(*x.shape) > self.rate) / (1.0 - self.rate)
            return x * self.mask
        return x
    def backward(self, dout):
        return dout * self.mask if (self.training and self.rate > 0) else dout


def softmax_cross_entropy_weighted(logits, y, class_weights):
    B     = len(y)
    shift = logits - logits.max(axis=1, keepdims=True)
    exp_s = np.exp(shift)
    probs = exp_s / exp_s.sum(axis=1, keepdims=True)
    log_p = np.log(probs[np.arange(B), y] + 1e-12)
    sw    = class_weights[y]
    loss  = -(sw * log_p).mean()
    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1
    dlogits *= sw[:, np.newaxis]
    dlogits /= B
    return loss, dlogits


class NumpyCNN:
    def __init__(self, n_features):
        self.conv1  = Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu1  = ReLU()
        self.conv2  = Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2  = ReLU()
        self.pool   = GlobalAvgPool1d()
        self.fc1    = Linear(32, 64)
        self.relu3  = ReLU()
        self.drop   = Dropout(0.3)
        self.fc2    = Linear(64, 2)
        self.layers = [
            self.conv1, self.relu1,
            self.conv2, self.relu2,
            self.pool,
            self.fc1, self.relu3, self.drop,
            self.fc2,
        ]

    def forward(self, x, training=True):
        out = x[:, np.newaxis, :]
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
            out = layer.forward(out)
        return out

    def backward(self, dlogits):
        g = dlogits
        for layer in reversed(self.layers):
            g = layer.backward(g)

    def predict(self, x):
        return self.forward(x, training=False).argmax(axis=1)

    def get_state(self):
        return {i: {'W': l.W.copy(), 'b': l.b.copy()}
                for i, l in enumerate(self.layers) if hasattr(l, 'W')}

    def set_state(self, state):
        for i, l in enumerate(self.layers):
            if i in state:
                l.W[:] = state[i]['W']
                l.b[:] = state[i]['b']


class Adam:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.t = 0; self._m = {}; self._v = {}

    def step(self, pairs):
        self.t += 1
        for key, (p, g) in enumerate(pairs):
            if key not in self._m:
                self._m[key] = np.zeros_like(p)
                self._v[key] = np.zeros_like(p)
            self._m[key] = self.b1 * self._m[key] + (1 - self.b1) * g
            self._v[key] = self.b2 * self._v[key] + (1 - self.b2) * g**2
            m_h = self._m[key] / (1 - self.b1**self.t)
            v_h = self._v[key] / (1 - self.b2**self.t)
            p  -= self.lr * m_h / (np.sqrt(v_h) + self.eps)


def train_numpy_cnn(X_train, y_train, X_val, y_val, n_features,
                    epochs=120, batch_size=64, lr=1e-3):
    """
    Train CNN on X_train / y_train.
    Use X_val / y_val ONLY for checkpoint selection — not reported as final accuracy.
    Returns (model, scaler) with best-val-acc weights restored.
    """
    scaler  = StandardScaler()
    Xtr     = scaler.fit_transform(X_train).astype(np.float64)
    Xva     = scaler.transform(X_val).astype(np.float64)

    counts        = np.bincount(y_train)
    class_weights = len(y_train) / (2.0 * counts)

    model     = NumpyCNN(n_features)
    optimizer = Adam(lr=lr)
    best_acc  = 0.0
    best_state = model.get_state()

    print(f"  Training CNN for {epochs} epochs  "
          f"(checkpoint selected on val, reported on test) ...")

    for epoch in range(1, epochs + 1):
        perm   = np.random.permutation(len(Xtr))
        Xtr_sh = Xtr[perm]
        ytr_sh = y_train[perm]
        ep_loss = 0.0; n_b = 0

        for start in range(0, len(Xtr_sh), batch_size):
            xb = Xtr_sh[start:start + batch_size]
            yb = ytr_sh[start:start + batch_size]
            logits        = model.forward(xb, training=True)
            loss, dlogits = softmax_cross_entropy_weighted(logits, yb, class_weights)
            model.backward(dlogits)
            pairs = [(l.W, l.dW) for l in model.layers if hasattr(l, 'W') and l.dW is not None]
            pairs += [(l.b, l.db) for l in model.layers if hasattr(l, 'W') and l.db is not None]
            optimizer.step(list(zip(
                [l.W for l in model.layers if hasattr(l,'W')] +
                [l.b for l in model.layers if hasattr(l,'W')],
                [l.dW for l in model.layers if hasattr(l,'W')] +
                [l.db for l in model.layers if hasattr(l,'W')]
            )))
            ep_loss += loss; n_b += 1

        # checkpoint selection on val (never reported as final accuracy)
        val_acc = accuracy_score(y_val, model.predict(Xva))
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = model.get_state()

        if epoch % 30 == 0:
            print(f"    Epoch {epoch:>3}/{epochs}  "
                  f"loss={ep_loss/n_b:.4f}  val_acc={val_acc:.3f}  best={best_acc:.3f}")

    model.set_state(best_state)
    print(f"  Best checkpoint val acc: {best_acc:.3f}  "
          f"(checkpoint selection only — final score uses test set)\n")
    return model, scaler


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred):
    return {
        "model":     name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1":        f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 62)
    print("  NBA GAME PREDICTION - MODEL COMPARISON (FIXED)")
    print("  SVM  vs  KNN  vs  CNN (numpy)")
    print("=" * 62)

    # ------------------------------------------------------------------
    # BUILD DATASET
    # ------------------------------------------------------------------
    print("\n[1/4] Building feature dataset ...\n")
    configure_nba_session()

    df_last = load_season_team_logs(LAST_SEASON)
    df_cur  = load_season_team_logs(CURRENT_SEASON)

    df_all          = pd.concat([df_last, df_cur], ignore_index=True)
    df_all          = add_elo_column(df_all)
    df_cur_with_elo = df_all[df_all["SEASON"] == CURRENT_SEASON].copy()

    last_avg   = compute_last_season_team_averages(df_last)
    df_cur_mom = add_current_season_momentum(df_cur_with_elo)
    df_blend   = blend_last_and_current(
        df_cur_mom, last_avg,
        w_last=W_LAST, w_cur=W_CUR, shrink_games=20,
    )

    X, y = build_game_level_examples(df_blend)
    feature_names = list(X.columns)
    n_features    = len(feature_names)

    print(f"\nDataset: {len(X)} games | {n_features} features | "
          f"home win rate: {y.mean():.1%}\n")

    # ------------------------------------------------------------------
    # THREE-WAY CHRONOLOGICAL SPLIT
    # build_game_level_examples sorts by GAME_DATE so row order = time order
    # ------------------------------------------------------------------
    n         = len(X)
    train_end = int(n * TRAIN_FRAC)
    val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))

    X_train = X.values[:train_end];     y_train = y.values[:train_end]
    X_val   = X.values[train_end:val_end]; y_val = y.values[train_end:val_end]
    X_test  = X.values[val_end:];       y_test  = y.values[val_end:]

    print(f"Chronological three-way split:")
    print(f"  Train : {len(X_train):>4} games  (rows   0 – {train_end-1})")
    print(f"  Val   : {len(X_val):>4} games  (rows {train_end} – {val_end-1})  "
          f"← CNN checkpoint selection only")
    print(f"  Test  : {len(X_test):>4} games  (rows {val_end} – {n-1})  "
          f"← final reported metrics for all models\n")

    results = []

    # ------------------------------------------------------------------
    # MODEL 1 — SVM
    # ------------------------------------------------------------------
    print("[2/4] Training SVM ...")
    svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=2.0, gamma="scale",
                    probability=True, class_weight="balanced")),
    ])
    svm_clf.fit(X_train, y_train)
    svm_pred = svm_clf.predict(X_test)
    results.append(evaluate("SVM (RBF, C=2)", y_test, svm_pred))
    print(f"  SVM test accuracy: {accuracy_score(y_test, svm_pred):.3f}\n")

    # ------------------------------------------------------------------
    # MODEL 2 — KNN  (TimeSeriesSplit for k selection)
    # ------------------------------------------------------------------
    print("[3/4] Training KNN ...")
    print("  k-selection via TimeSeriesSplit(n_splits=5) on training set")
    print("  (ensures no fold trains on future games and tests on past games)\n")

    tscv = TimeSeriesSplit(n_splits=5)
    best_k, best_cv = 5, 0.0

    for k in [3, 5, 7, 11, 15, 21]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(
                n_neighbors=k, weights="distance", metric="euclidean")),
        ])
        cv = cross_val_score(
            pipe, X_train, y_train,
            cv=tscv,                # time-aware folds
            scoring="accuracy",
        ).mean()
        print(f"  k={k:>2}  TimeSeriesCV acc = {cv:.3f}")
        if cv > best_cv:
            best_cv, best_k = cv, k

    print(f"\n  Best k={best_k}  (CV acc={best_cv:.3f})")
    knn_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=best_k, weights="distance", metric="euclidean")),
    ])
    knn_clf.fit(X_train, y_train)
    knn_pred = knn_clf.predict(X_test)
    results.append(evaluate(f"KNN (k={best_k})", y_test, knn_pred))
    print(f"  KNN test accuracy: {accuracy_score(y_test, knn_pred):.3f}\n")

    # ------------------------------------------------------------------
    # MODEL 3 — CNN
    # Trained on X_train, checkpoint selected on X_val, reported on X_test
    # ------------------------------------------------------------------
    print("[4/4] Training CNN (pure numpy) ...")
    np.random.seed(RANDOM_STATE)
    cnn_model, cnn_scaler = train_numpy_cnn(
        X_train, y_train,
        X_val,   y_val,      # val used for checkpoint only
        n_features=n_features,
        epochs=120,
        batch_size=64,
        lr=1e-3,
    )
    # report on test — never seen during training or checkpoint selection
    cnn_pred = cnn_model.predict(cnn_scaler.transform(X_test))
    results.append(evaluate("CNN (1D conv)", y_test, cnn_pred))
    print(f"  CNN test accuracy: {accuracy_score(y_test, cnn_pred):.3f}\n")

    # ------------------------------------------------------------------
    # COMPARISON TABLE
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    sep        = "=" * 62

    print(sep)
    print("  FINAL COMPARISON — all scores on held-out TEST set")
    print("  (weighted avg precision / recall / F1)")
    print(sep)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'-'*58}")
    best_acc = results_df["accuracy"].max()
    for _, row in results_df.iterrows():
        tag = "  <-- best" if row["accuracy"] == best_acc else ""
        print(
            f"  {row['model']:<22}"
            f"{row['accuracy']:>8.3f}  "
            f"{row['precision']:>9.3f}  "
            f"{row['recall']:>7.3f}  "
            f"{row['f1']:>7.3f}"
            f"{tag}"
        )
    print(sep)

    print("\n--- Detailed Classification Reports ---\n")
    for name, preds in [
        ("SVM", svm_pred),
        (f"KNN (k={best_k})", knn_pred),
        ("CNN (numpy)", cnn_pred),
    ]:
        print(f"  {name}:")
        print(classification_report(
            y_test, preds,
            target_names=["Away Win", "Home Win"],
        ))

    # save
    out_csv = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Results saved to: {out_csv.resolve()}")

    joblib.dump({"model": svm_clf, "feature_names": feature_names},
                OUTPUT_DIR / "model_svm.pkl")
    joblib.dump({"model": knn_clf, "feature_names": feature_names},
                OUTPUT_DIR / "model_knn.pkl")
    joblib.dump({"model": cnn_model, "scaler": cnn_scaler,
                 "feature_names": feature_names},
                OUTPUT_DIR / "model_cnn.pkl")
    print("All models saved to outputs/")
    print(sep)


if __name__ == "__main__":
    main()
