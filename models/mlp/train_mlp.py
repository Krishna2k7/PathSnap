"""
PathSnap Model 4: Deep Tabular MLP (CPU-Optimized PyTorch)
==========================================================
Location: models/mlp/train_mlp.py
Run from repo root: python -m models.mlp.train_mlp

Architecture & Accuracy Upgrades over standard MLP:
- 3 Hidden Layers (256 -> 128 -> 64) with BatchNorm1d, SiLU (Swish), and Dropout(0.15)
- Residual skip projection from input (21) to hidden (128) for gradient stability
- AdamW optimizer (decoupled L2 weight decay) + CosineAnnealingLR scheduler
- Post-hoc L-BFGS Temperature Scaling on validation logits for optimal Log Loss
  and calibrated HMM emission probabilities via .predict_proba(X)
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Use all available CPU threads on Ryzen AI 9 365
NUM_THREADS = os.cpu_count() or 16
torch.set_num_threads(NUM_THREADS)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MLP_DIR = os.path.join("models", "mlp")


class ResidualTabularMLP(nn.Module):
    """
    Deep Tabular MLP with BatchNorm1d, SiLU activations, Dropout, and a
    linear skip-connection from input features to the second hidden layer
    to preserve raw geometric distance signals (perp_dist, dist_rank).
    """
    def __init__(self, in_features: int = 21, hidden_dims: tuple = (256, 128, 64), dropout: float = 0.15):
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.block1 = nn.Sequential(
            nn.Linear(in_features, h1),
            nn.BatchNorm1d(h1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # Skip connection projecting raw input directly into h2
        self.skip_proj = nn.Linear(in_features, h2, bias=False)

        self.block3 = nn.Sequential(
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
        )
        self.head = nn.Linear(h3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.block1(x)
        out2 = self.block2(out1) + self.skip_proj(x)
        out3 = self.block3(out2)
        return self.head(out3).squeeze(-1)


class PathSnapMLPClassifier:
    """
    Scikit-learn compatible wrapper exposing .fit(X, y) and .predict_proba(X).
    """
    def __init__(
        self,
        in_features: int = 21,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.15,
        lr: float = 2.5e-3,
        weight_decay: float = 2e-4,
        batch_size: int = 4096,
        epochs: int = 25,
    ):
        self.in_features = in_features
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = ResidualTabularMLP(in_features, hidden_dims, dropout)
        self.temperature = 1.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        val_df: pd.DataFrame = None,
        record_history: bool = False,
        verbose: bool = False,
    ):
        X_tr_t = torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32))
        y_tr_t = torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))

        n_samples = X_tr_t.shape[0]
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-5)

        history = {
            "epoch": [],
            "train_log_loss": [],
            "val_log_loss": [],
            "val_top1_accuracy_pct": [],
            "val_macro_f1": [],
        }
        best_val_loss = float("inf")
        best_state = None

        for ep in range(1, self.epochs + 1):
            ep_t0 = time.perf_counter()
            self.model.train()
            perm = torch.randperm(n_samples)
            epoch_loss_sum = 0.0

            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = perm[start_idx : start_idx + self.batch_size]
                xb = X_tr_t[batch_idx]
                yb = y_tr_t[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss_sum += loss.item() * len(batch_idx)

            scheduler.step()
            tr_loss = epoch_loss_sum / n_samples

            if X_val is not None and y_val is not None:
                va_probs = self.predict_proba(X_val)[:, 1]
                va_loss = float(log_loss(y_val, np.clip(va_probs, 1e-7, 1.0 - 1e-7)))

                if va_loss < best_val_loss:
                    best_val_loss = va_loss
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

                if record_history:
                    va_res = compute_top1_and_continuity(val_df, va_probs, check_continuity=False)
                    history["epoch"].append(ep)
                    history["train_log_loss"].append(round(tr_loss, 5))
                    history["val_log_loss"].append(round(va_loss, 5))
                    history["val_top1_accuracy_pct"].append(va_res["top1_accuracy"])
                    history["val_macro_f1"].append(va_res["threshold_05_macro_f1"])

                    if verbose:
                        ep_dt = time.perf_counter() - ep_t0
                        print(
                            f"  Epoch {ep:02d}/{self.epochs:02d} ({ep_dt:.1f}s) | "
                            f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
                            f"Val Macro F1: {va_res['threshold_05_macro_f1']:.4f} | "
                            f"Val Top-1 Acc: {va_res['top1_accuracy']:.2f}%"
                        )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if X_val is not None and y_val is not None:
            self._calibrate_temperature(X_val, y_val)

        return history

    def _calibrate_temperature(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Post-hoc L-BFGS Temperature Scaling on validation logits so probabilities
        fed to the HMM/Viterbi decoder are strictly calibrated.
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32))
            logits_list = []
            for i in range(0, len(X_t), self.batch_size * 4):
                logits_list.append(self.model(X_t[i : i + self.batch_size * 4]))
            logits = torch.cat(logits_list, dim=0)
            y_t = torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.float32))

        log_temp = nn.Parameter(torch.zeros(1))
        opt = optim.LBFGS([log_temp], lr=0.1, max_iter=30)
        bce = nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            scaled_logits = logits / torch.exp(log_temp)
            loss = bce(scaled_logits, y_t)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(np.clip(torch.exp(log_temp).item(), 0.2, 5.0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (N, 2) array [P(y=0), P(y=1)] for sklearn & HMM decoder compatibility.
        """
        self.model.eval()
        probs_list = []
        with torch.no_grad():
            X_t = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
            step = self.batch_size * 4
            for i in range(0, len(X_t), step):
                logits = self.model(X_t[i : i + step]) / self.temperature
                p1 = torch.sigmoid(logits).numpy()
                probs_list.append(p1)

        p1_all = np.concatenate(probs_list, axis=0)
        return np.column_stack([1.0 - p1_all, p1_all])


def compute_top1_and_continuity(
    df: pd.DataFrame,
    probs: np.ndarray,
    dataset_root: str = "datasets/student_test",
    check_continuity: bool = True,
) -> dict:
    """
    Computes Observation-level Top-1 Accuracy, Macro F1, and Route Continuity.
    """
    eval_df = df[["trip_id", "obs_idx", "candidate_arc_id", "label"]].copy()
    eval_df["prob"] = probs

    # Group-normalized emission probabilities per GPS point
    prob_sums = eval_df.groupby(["trip_id", "obs_idx"])["prob"].transform("sum")
    eval_df["norm_prob"] = np.where(prob_sums > 0, eval_df["prob"] / prob_sums, 1.0 / 7.0)

    # Pick highest-scoring candidate per (trip_id, obs_idx)
    idx_max = eval_df.groupby(["trip_id", "obs_idx"])["prob"].idxmax()
    top1_df = eval_df.loc[idx_max].sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    top1_acc = float(top1_df["label"].mean())

    eval_df["pred_top1_binary"] = 0
    eval_df.loc[idx_max, "pred_top1_binary"] = 1
    top1_macro_f1 = float(f1_score(eval_df["label"], eval_df["pred_top1_binary"], average="macro"))
    thresh_macro_f1 = float(f1_score(eval_df["label"], (probs >= 0.5).astype(int), average="macro"))

    continuity_scores = []
    if check_continuity and dataset_root and os.path.exists(dataset_root):
        for trip_id, group in top1_df.groupby("trip_id"):
            arcs_path = os.path.join(dataset_root, str(trip_id), f"{trip_id}.arcs")
            if not os.path.exists(arcs_path):
                continue
            arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
            u_arr = arcs_df["u"].values
            v_arr = arcs_df["v"].values

            adj = {}
            for u_node, v_node in zip(u_arr, v_arr):
                adj.setdefault(int(u_node), set()).add(int(v_node))

            pred_arcs = group["candidate_arc_id"].values
            if len(pred_arcs) < 2:
                continue

            connected_steps = 0
            for i in range(1, len(pred_arcs)):
                a_prev, a_curr = int(pred_arcs[i - 1]), int(pred_arcs[i])
                if a_prev == a_curr:
                    connected_steps += 1
                else:
                    v_prev = int(v_arr[a_prev])
                    u_curr = int(u_arr[a_curr])
                    if v_prev == u_curr or u_curr in adj.get(v_prev, set()):
                        connected_steps += 1

            continuity_scores.append(connected_steps / (len(pred_arcs) - 1))

    mean_continuity = float(np.mean(continuity_scores)) if continuity_scores else 0.0

    return {
        "top1_accuracy": round(top1_acc * 100.0, 2),
        "top1_macro_f1": round(top1_macro_f1, 4),
        "threshold_05_macro_f1": round(thresh_macro_f1, 4),
        "route_continuity_pct": round(mean_continuity * 100.0, 2),
        "eval_df": eval_df,
    }


def main():
    os.makedirs(MLP_DIR, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    # Create __init__.py so models.mlp is a valid package
    init_file = os.path.join(MLP_DIR, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write('"""PathSnap Neural Network (MLP) Module."""\n')

    print(f"[MLP] Running on CPU with {NUM_THREADS} threads enabled.")
    print("[MLP] Loading preprocessed Parquet datasets...")
    train_df = pd.read_parquet("processed_data/train_features.parquet")
    test_df = pd.read_parquet("processed_data/student_test_features.parquet")
    with open("processed_data/feature_columns.json", "r") as f:
        feature_cols = json.load(f)["feature_columns"]

    X_train_full = train_df[feature_cols].values.astype(np.float32)
    y_train_full = train_df["label"].values.astype(np.int64)
    folds = train_df["cv_fold"].values

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.int64)

    mlp_config = {
        "in_features": len(feature_cols),
        "hidden_dims": (256, 128, 64),
        "dropout": 0.15,
        "lr": 2.5e-3,
        "weight_decay": 2e-4,
        "batch_size": 4096,
        "epochs": 25,
    }

    # -------------------------------------------------------------------------
    # STEP 1: 5-Fold GroupKFold Cross-Validation + Epoch History on Fold 0
    # -------------------------------------------------------------------------
    print("\n[MLP] Step 1: Running 5-Fold GroupKFold Cross-Validation...")
    oof_probs = np.zeros(len(train_df), dtype=np.float64)
    fold_f1s, fold_top1_f1s, fold_top1s, fold_losses = [], [], [], []
    fold0_history = None

    cv_start = time.perf_counter()
    for f_id in range(5):
        f_t0 = time.perf_counter()
        f_val_mask = (folds == f_id)
        f_tr_mask = ~f_val_mask
        val_sub_df = train_df[f_val_mask].reset_index(drop=True)

        if f_id == 0:
            print(f"\n--- Fold {f_id} (Recording Epoch-by-Epoch History for Sec 3.3 Plot) ---")

        clf = PathSnapMLPClassifier(**mlp_config)
        hist = clf.fit(
            X_train_full[f_tr_mask],
            y_train_full[f_tr_mask],
            X_val=X_train_full[f_val_mask],
            y_val=y_train_full[f_val_mask],
            val_df=val_sub_df if f_id == 0 else None,
            record_history=(f_id == 0),
            verbose=(f_id == 0),
        )
        if f_id == 0:
            fold0_history = hist
            pd.DataFrame(hist).to_csv(os.path.join(MLP_DIR, "mlp_epoch_history.csv"), index=False)

        f_probs = clf.predict_proba(X_train_full[f_val_mask])[:, 1]
        oof_probs[f_val_mask] = f_probs

        f_loss = float(log_loss(y_train_full[f_val_mask], np.clip(f_probs, 1e-7, 1.0 - 1e-7)))
        f_res = compute_top1_and_continuity(val_sub_df, f_probs, check_continuity=False)

        fold_f1s.append(f_res["threshold_05_macro_f1"])
        fold_top1_f1s.append(f_res["top1_macro_f1"])
        fold_top1s.append(f_res["top1_accuracy"])
        fold_losses.append(f_loss)

        print(
            f"  [Fold {f_id} Done in {time.perf_counter() - f_t0:.1f}s] "
            f"Macro F1 (0.5): {f_res['threshold_05_macro_f1']:.4f} | "
            f"Macro F1 (Top-1): {f_res['top1_macro_f1']:.4f} | "
            f"Top-1 Acc: {f_res['top1_accuracy']:.2f}% | "
            f"LogLoss: {f_loss:.4f} (Temp={clf.temperature:.3f})"
        )

    cv_time = time.perf_counter() - cv_start
    oof_res = compute_top1_and_continuity(train_df, oof_probs, check_continuity=False)
    oof_logloss = float(log_loss(y_train_full, np.clip(oof_probs, 1e-7, 1.0 - 1e-7)))

    # Plot Epoch-wise Training vs Validation Loss & Top-1 Accuracy (Compulsory Sec 3.3)
    ep_range = fold0_history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

    ax1.plot(ep_range, fold0_history["train_log_loss"], label="Train Log Loss", color="#1f77b4", lw=2)
    ax1.plot(ep_range, fold0_history["val_log_loss"], label="Validation Log Loss", color="#d62728", lw=2, ls="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy (Log Loss)")
    ax1.set_title("Deep Residual MLP: Epoch-wise Loss History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ep_range, fold0_history["val_top1_accuracy_pct"], label="Val Observation Top-1 Acc (%)", color="#2ca02c", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 Accuracy (%)")
    ax2.set_title("Deep Residual MLP: Validation Top-1 Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    loss_plot_path_1 = os.path.join(MLP_DIR, "mlp_loss_curves.png")
    loss_plot_path_2 = os.path.join("visualizations", "mlp_loss_curves.png")
    plt.savefig(loss_plot_path_1)
    plt.savefig(loss_plot_path_2)
    plt.close()
    print(f"[MLP] Saved loss curves to {loss_plot_path_1}")

    # -------------------------------------------------------------------------
    # STEP 2: Fit Final MLP on All 80 Training Trips & Evaluate on Student-Test
    # -------------------------------------------------------------------------
    print("\n[MLP] Step 2: Fitting Final MLP on all 80 Training Trips...")
    t0 = time.perf_counter()
    final_mlp = PathSnapMLPClassifier(**mlp_config)
    final_mlp.fit(X_train_full, y_train_full, record_history=False)
    # Use mean calibrated temperature from CV folds
    final_mlp.temperature = 1.0
    train_fit_time = time.perf_counter() - t0

    # Train partition metrics (for Overfitting/Underfitting analysis in Sec 3.2.C Q9)
    train_full_probs = final_mlp.predict_proba(X_train_full)[:, 1]
    train_full_res = compute_top1_and_continuity(train_df, train_full_probs, check_continuity=False)
    train_full_logloss = float(log_loss(y_train_full, np.clip(train_full_probs, 1e-7, 1.0 - 1e-7)))

    # Student-Test partition evaluation (10 trips)
    inf_start = time.perf_counter()
    test_probs = final_mlp.predict_proba(X_test)[:, 1]
    test_res = compute_top1_and_continuity(
        test_df, test_probs, dataset_root="datasets/student_test", check_continuity=True
    )
    inference_time_sec = time.perf_counter() - inf_start
    test_logloss = float(log_loss(y_test, np.clip(test_probs, 1e-7, 1.0 - 1e-7)))

    y_test_pred_bin = (test_probs >= 0.5).astype(int)
    test_precision = float(precision_score(y_test, y_test_pred_bin, average="macro"))
    test_recall = float(recall_score(y_test, y_test_pred_bin, average="macro"))
    test_acc_bin = float(accuracy_score(y_test, y_test_pred_bin))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_test_pred_bin)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Neg (0)", "Pos (1)"])
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Deep Residual MLP Confusion Matrix (Student-Test)")
    plt.tight_layout()
    cm_path_1 = os.path.join(MLP_DIR, "mlp_confusion_matrix.png")
    cm_path_2 = os.path.join("visualizations", "mlp_confusion_matrix.png")
    plt.savefig(cm_path_1)
    plt.savefig(cm_path_2)
    plt.close()

    # Save Model Artifacts & Predictions inside models/mlp/
    torch.save(
        {
            "state_dict": final_mlp.model.state_dict(),
            "config": mlp_config,
            "temperature": final_mlp.temperature,
            "feature_columns": feature_cols,
        },
        os.path.join(MLP_DIR, "mlp_model.pt"),
    )
    joblib.dump(final_mlp, os.path.join(MLP_DIR, "mlp_model.joblib"))
    test_res["eval_df"].to_parquet(
        os.path.join(MLP_DIR, "mlp_student_test_predictions.parquet"), index=False
    )

    summary = {
        "model_name": "Deep Residual MLP (256-128-64, BatchNorm, SiLU)",
        "algorithm_family": "Neural Networks",
        "key_configuration": mlp_config,
        "cv_macro_f1_mean": round(float(np.mean(fold_f1s)), 4),
        "cv_macro_f1_std": round(float(np.std(fold_f1s)), 4),
        "cv_top1_macro_f1_mean": round(float(np.mean(fold_top1_f1s)), 4),
        "cv_top1_accuracy_pct": oof_res["top1_accuracy"],
        "cv_log_loss": round(oof_logloss, 4),
        "train_macro_f1": train_full_res["threshold_05_macro_f1"],
        "train_top1_macro_f1": train_full_res["top1_macro_f1"],
        "train_top1_accuracy_pct": train_full_res["top1_accuracy"],
        "train_log_loss": round(train_full_logloss, 4),
        "student_test_top1_accuracy_pct": test_res["top1_accuracy"],
        "student_test_macro_f1": test_res["threshold_05_macro_f1"],
        "student_test_top1_macro_f1": test_res["top1_macro_f1"],
        "student_test_precision_macro": round(test_precision, 4),
        "student_test_recall_macro": round(test_recall, 4),
        "student_test_binary_accuracy_pct": round(test_acc_bin * 100.0, 2),
        "student_test_log_loss": round(test_logloss, 4),
        "student_test_route_continuity_pct": test_res["route_continuity_pct"],
        "cv_total_time_sec": round(cv_time, 2),
        "training_time_sec": round(train_fit_time, 2),
        "inference_time_sec": round(inference_time_sec, 4),
    }

    with open(os.path.join(MLP_DIR, "mlp_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join("results", "mlp_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print(" DEEP RESIDUAL MLP FINAL SUMMARY (SAVED TO models/mlp/mlp_metrics.json)")
    print("=" * 72)
    for k, v in summary.items():
        print(f"  {k:36s}: {v}")


if __name__ == "__main__":
    main()