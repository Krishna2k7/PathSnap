import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    log_loss, confusion_matrix
)
def load_arc_topology(test_dir="datasets/student_test"):
    topo = {}
    for trip_id in os.listdir(test_dir):
        arcs_path = os.path.join(test_dir, trip_id, f"{trip_id}.arcs")
        if os.path.exists(arcs_path):
            arcs = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"])
            topo[trip_id] = (arcs["u"].values, arcs["v"].values)
    return topo
def compute_top1_and_continuity(test_df, probs, arc_topo):
    df = test_df[["trip_id", "obs_idx", "candidate_arc_id", "label"]].copy()
    df["prob"] = probs
    top1 = df.sort_values("prob", ascending=False).groupby(["trip_id", "obs_idx"], as_index=False).first()
    top1 = top1.sort_values(["trip_id", "obs_idx"])
    top1_acc = float(top1["label"].mean())
    valid_transitions, total_transitions = 0, 0
    for trip_id, grp in top1.groupby("trip_id"):
        if trip_id not in arc_topo:
            continue
        u_arr, v_arr = arc_topo[trip_id]
        pred_arcs = grp["candidate_arc_id"].values
        for i in range(len(pred_arcs) - 1):
            a1, a2 = int(pred_arcs[i]), int(pred_arcs[i + 1])
            total_transitions += 1
            if (a1 == a2) or (v_arr[a1] == u_arr[a2]) or (u_arr[a1] == v_arr[a2]):
                valid_transitions += 1
    continuity = float(valid_transitions / max(total_transitions, 1))
    return top1_acc, continuity
def stratified_subsample(X, y, max_rows, seed=42):
    if len(X) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_ratio = len(pos_idx) / len(y)
    n_pos = max(1, int(round(max_rows * pos_ratio)))
    n_neg = max_rows - n_pos
    chosen = np.concatenate([
        rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    ])
    rng.shuffle(chosen)
    return X[chosen], y[chosen]
def build_model(model_name):
    if model_name == "Distance_Weighted_KNN":
        return KNeighborsClassifier(
            n_neighbors=35,
            weights="distance",
            metric="minkowski",
            p=2,
            leaf_size=60,
            n_jobs=-1
        )
    elif model_name == "RBF_SVM":
        return SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    raise ValueError(f"Unknown model: {model_name}")
if __name__ == "__main__":
    os.makedirs("results/model5", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)
    with open("processed_data/feature_columns.json") as f:
        feature_cols = json.load(f)["feature_columns"]
    print(f"Loading processed parquet datasets ({len(feature_cols)} features)...", flush=True)
    train_df = pd.read_parquet("processed_data/train_features.parquet")
    test_df = pd.read_parquet("processed_data/student_test_features.parquet")
    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.int32)
    cv_folds = train_df["cv_fold"].to_numpy(dtype=np.int32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.int32)
    arc_topo = load_arc_topology("datasets/student_test")
    model_names = ["Distance_Weighted_KNN", "RBF_SVM"]
    results = []
    for name in model_names:
        print(f"\n==========================================", flush=True)
        print(f" Training & Evaluating: {name}", flush=True)
        print(f"==========================================", flush=True)
        cap_cv = 120000 if name == "Distance_Weighted_KNN" else 18000
        cap_full = 180000 if name == "Distance_Weighted_KNN" else 22000
        cv_f1s = []
        for fold in sorted(np.unique(cv_folds)):
            t_fold = time.perf_counter()
            val_mask = (cv_folds == fold)
            tr_mask = ~val_mask
            X_tr_f, y_tr_f = stratified_subsample(X_train[tr_mask], y_train[tr_mask], max_rows=cap_cv, seed=42 + fold)
            X_va_f, y_va_f = X_train[val_mask], y_train[val_mask]
            clf = build_model(name)
            clf.fit(X_tr_f, y_tr_f)
            val_preds = clf.predict(X_va_f)
            fold_f1 = f1_score(y_va_f, val_preds, average="macro")
            cv_f1s.append(fold_f1)
            print(f"  [CV Fold {fold}] Macro F1 = {fold_f1:.4f} ({time.perf_counter() - t_fold:.1f}s)", flush=True)
        X_full, y_full = stratified_subsample(X_train, y_train, max_rows=cap_full, seed=42)
        print(f"  [Full Fit] Fitting on {len(X_full):,} stratified training rows...", flush=True)
        final_clf = build_model(name)
        t0 = time.perf_counter()
        final_clf.fit(X_full, y_full)
        train_time = time.perf_counter() - t0
        t1 = time.perf_counter()
        test_probs = final_clf.predict_proba(X_test)[:, 1]
        infer_time = time.perf_counter() - t1
        test_preds = (test_probs >= 0.5).astype(int)
        top1_acc, continuity = compute_top1_and_continuity(test_df, test_probs, arc_topo)
        cm = confusion_matrix(y_test, test_preds)
        row = {
            "Model": name,
            "Algorithm_Family": "Instance-Based" if "KNN" in name else "Kernel-Based",
            "Key_Configuration": "k=35, weights=distance, Minkowski p=2, leaf_size=60" if "KNN" in name else "RBF kernel, C=1.0, gamma=scale, probability=True, balanced",
            "CV_Macro_F1_Mean": round(float(np.mean(cv_f1s)), 4),
            "CV_Macro_F1_Std": round(float(np.std(cv_f1s)), 4),
            "Top1_Accuracy": round(top1_acc, 4),
            "Route_Continuity": round(continuity, 4),
            "Student_Test_Macro_F1": round(float(f1_score(y_test, test_preds, average="macro")), 4),
            "Precision_Macro": round(float(precision_score(y_test, test_preds, average="macro", zero_division=0)), 4),
            "Recall_Macro": round(float(recall_score(y_test, test_preds, average="macro", zero_division=0)), 4),
            "Log_Loss": round(float(log_loss(y_test, np.clip(test_probs, 1e-6, 1 - 1e-6))), 4),
            "Train_Time_s": round(train_time, 3),
            "Inference_Time_s": round(infer_time, 3),
            "Confusion_Matrix": cm.tolist(),
        }
        results.append(row)
        joblib.dump(final_clf, f"saved_models/{name.lower()}.joblib")
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg (0)", "Pos (1)"], yticklabels=["Neg (0)", "Pos (1)"])
        plt.title(f"Confusion Matrix: {name} (Student-Test)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(f"results/model5/{name.lower()}_confusion_matrix.png", dpi=150)
        plt.close()
    summary_df = pd.DataFrame(results)
    summary_df.to_csv("results/model5/model5_comparison.csv", index=False)
    with open("results/model5/model5_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== INSTANCE-BASED & KERNEL-BASED EXPERIMENTATION SUMMARY ===")
    print(summary_df.drop(columns=["Confusion_Matrix", "Key_Configuration"]).to_string(index=False))
