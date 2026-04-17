"""
05_deep_baselines.py
====================
Phase 2, Step 2 — LSTM and 1D-CNN deep-learning baselines, matching what
Reviewer 2 asked for ("LSTM, temporal boosting, or Transformer-based
regressors") and Reviewer 1 ("recurrent neural networks, temporal
convolutional networks").

Uses the SAME features as the RF/XGBoost scripts, but reshapes the
15-dim feature vector into:
  - sequence input:  11 timesteps x 1 feature  (SNR_Lag_10 ... SNR_Lag_1, SNR_t)
  - auxiliary input: 4 values (Velocity, Acceleration, Rolling_Std, Rolling_Mean)

Both models go:
    seq -> (LSTM|Conv1D) -> hidden vector -> concat with aux -> FC -> TTF

Run:
    python 05_deep_baselines.py \
        --features_dir real_data_features \
        --output_dir real_data_results_deep \
        --models lstm cnn \
        --n_seeds 3 \
        --epochs 30 --batch_size 4096 \
        --train_pkl real_data_processed/train_full.pkl

NOTES:
  - With ~1.75 M training rows, GPU is strongly recommended.
    Detected automatically; falls back to CPU with warning.
  - On CPU, reduce load via:  --max_train 500000
    (random-subsamples training set; val/test untouched).
  - Early stopping on val MAE with patience 4.

Dependencies:
    pip install torch
"""

import argparse
import json
import time
import sys
import warnings
from pathlib import Path
import importlib.util as _ilu

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# region agent log
def _agent_log(message, data, *, run_id="pre-fix", hypothesis_id="A"):
    payload = {
        "sessionId": "0dbeec",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "05_deep_baselines.py:imports",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("debug-0dbeec.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# endregion

try:
    # region agent log
    _agent_log(
        "attempting_torch_import",
        {"python_executable": sys.executable, "torch_spec": str(_ilu.find_spec("torch"))},
        hypothesis_id="A",
    )
    # endregion
    import torch
    import torch.nn as nn
    # region agent log
    _agent_log(
        "before_import_torch_utils_data",
        {"python_executable": sys.executable, "torch_utils_data_spec": str(_ilu.find_spec("torch.utils.data"))},
        hypothesis_id="B",
    )
    # endregion
    from torch.utils.data import DataLoader, TensorDataset
    # region agent log
    _agent_log(
        "torch_import_ok",
        {"python_executable": sys.executable, "torch_version": getattr(torch, "__version__", None)},
        hypothesis_id="A",
    )
    # endregion
except ImportError:
    # region agent log
    _agent_log(
        "torch_import_failed",
        {"python_executable": sys.executable, "torch_spec": str(_ilu.find_spec("torch"))},
        hypothesis_id="A",
    )
    # endregion
    raise SystemExit("PyTorch not installed. Run:  pip install torch\n"
                     "If this installs CPU-only and you have a GPU, see "
                     "https://pytorch.org/get-started/locally/ for the "
                     "correct CUDA-enabled install command.")


# ----------------------------------------------------------------------
# Feature reshaping (identical for both models)
# ----------------------------------------------------------------------
def reshape_features(X, feature_names):
    """
    Convert (N, 15) feature matrix into:
        X_seq: (N, 11, 1)  - SNR_Lag_10, ..., SNR_Lag_1, SNR_dB (oldest -> newest)
        X_aux: (N, 4)      - Velocity, Acceleration, Rolling_Std, Rolling_Mean
    """
    # Identify columns
    snr_idx = feature_names.index("SNR_dB")
    lag_idx = [feature_names.index(f"SNR_Lag_{i}") for i in range(1, 11)]
    vel_idx = feature_names.index("Velocity")
    acc_idx = feature_names.index("Acceleration")
    rstd_idx = feature_names.index("Rolling_Std")
    rmean_idx = feature_names.index("Rolling_Mean")

    # Build sequence: oldest (lag_10) first, newest (SNR_dB) last
    seq_order = lag_idx[::-1] + [snr_idx]   # Lag_10, Lag_9, ..., Lag_1, current
    X_seq = X[:, seq_order][..., np.newaxis]    # (N, 11, 1)
    X_aux = X[:, [vel_idx, acc_idx, rstd_idx, rmean_idx]]   # (N, 4)
    return X_seq.astype(np.float32), X_aux.astype(np.float32)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class LSTMPredictor(nn.Module):
    def __init__(self, hidden=64, num_layers=2, aux_dim=4, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + aux_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq, aux):
        _, (h, _) = self.lstm(seq)          # h: (num_layers, N, hidden)
        h_last = h[-1]                      # (N, hidden)
        return self.head(torch.cat([h_last, aux], dim=1)).squeeze(-1)


class CNN1DPredictor(nn.Module):
    def __init__(self, channels=(32, 64, 64), aux_dim=4, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1] + aux_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, seq, aux):
        # seq: (N, L=11, F=1)  ->  (N, F, L) for Conv1d
        x = seq.permute(0, 2, 1)
        h = self.conv(x).squeeze(-1)        # (N, channels[-1])
        return self.head(torch.cat([h, aux], dim=1)).squeeze(-1)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_one_model(model, train_loader, val_loader, epochs, lr, device, patience=4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()       # optimize MAE directly
    model = model.to(device)

    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for seq, aux, y in train_loader:
            seq, aux, y = seq.to(device), aux.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(seq, aux)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        train_mae = tot / n

        # Validation
        model.eval()
        v_tot, v_n = 0.0, 0
        with torch.no_grad():
            for seq, aux, y in val_loader:
                seq, aux, y = seq.to(device), aux.to(device), y.to(device)
                pred = model(seq, aux)
                v_tot += loss_fn(pred, y).item() * y.size(0)
                v_n += y.size(0)
        val_mae = v_tot / v_n
        dt = time.time() - t0
        print(f"    epoch {ep:2d}/{epochs}  train MAE={train_mae:.2f}  "
              f"val MAE={val_mae:.2f}  time={dt:.1f}s", flush=True)

        if val_mae < best_val - 1e-3:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    early stop (no val improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict_model(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for seq, aux, _y in loader:
            seq, aux = seq.to(device), aux.to(device)
            preds.append(model(seq, aux).cpu().numpy())
    return np.concatenate(preds)


# ----------------------------------------------------------------------
# Evaluation helpers (same contract as earlier scripts)
# ----------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, approaching_mask=None):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except ValueError:
        r2 = float("nan")
    result = {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}
    if approaching_mask is not None and approaching_mask.sum() > 0:
        yt, yp = y_true[approaching_mask], y_pred[approaching_mask]
        result["MAE_approaching"] = float(mean_absolute_error(yt, yp))
        result["RMSE_approaching"] = float(np.sqrt(mean_squared_error(yt, yp)))
        result["R2_approaching"] = float(r2_score(yt, yp)) if yt.std() > 0 else float("nan")
    return result


def per_class_metrics(y_true, y_pred, traj_test, stats_df, cap):
    rows = []
    for lbl in [0, 1, 2, 3]:
        class_tids = stats_df[stats_df["label"] == lbl]["trajectory_id"].values
        mask_all = np.isin(traj_test, class_tids)
        if mask_all.sum() == 0:
            continue
        yt, yp = y_true[mask_all], y_pred[mask_all]
        row = {
            "label":         lbl,
            "n_samples":     int(mask_all.sum()),
            "n_approaching": int((yt < cap).sum()),
            "MAE_all":       float(mean_absolute_error(yt, yp)),
            "RMSE_all":      float(np.sqrt(mean_squared_error(yt, yp))),
            "R2_all":        float(r2_score(yt, yp)) if yt.std() > 0 else float("nan"),
        }
        sub = yt < cap
        if sub.sum() > 0:
            row["MAE_approaching"]  = float(mean_absolute_error(yt[sub], yp[sub]))
            row["RMSE_approaching"] = float(np.sqrt(mean_squared_error(yt[sub], yp[sub])))
            row["R2_approaching"]   = float(r2_score(yt[sub], yp[sub])) \
                                        if yt[sub].std() > 0 else float("nan")
        else:
            row["MAE_approaching"]  = float("nan")
            row["RMSE_approaching"] = float("nan")
            row["R2_approaching"]   = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="real_data_features")
    parser.add_argument("--output_dir",   default="real_data_results_deep")
    parser.add_argument("--models", nargs="+", default=["lstm", "cnn"],
                        choices=["lstm", "cnn"])
    parser.add_argument("--n_seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max_train", type=int, default=None,
                        help="Randomly subsample training set to this size (CPU survival)")
    parser.add_argument("--train_pkl", default=None)
    args = parser.parse_args()

    feat_dir = Path(args.features_dir)
    out_dir = Path(args.output_dir)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  (CUDA available: {torch.cuda.is_available()})")
    if device.type == "cpu":
        warnings.warn("Training LSTM/CNN on CPU is slow. "
                      "Consider --max_train 500000 if this becomes painful.")

    # -------- Load and reshape --------
    print("Loading features ...")
    X_tr_raw = np.load(feat_dir / "X_train.npy")
    y_tr     = np.load(feat_dir / "y_train.npy")
    X_va_raw = np.load(feat_dir / "X_val.npy")
    y_va     = np.load(feat_dir / "y_val.npy")
    X_te_raw = np.load(feat_dir / "X_test.npy")
    y_te     = np.load(feat_dir / "y_test.npy")
    traj_te  = np.load(feat_dir / "traj_test.npy")
    feature_names = json.loads((feat_dir / "feature_names.json").read_text())
    cap = float((feat_dir / "ttf_cap.txt").read_text().strip())

    # Subsample train if requested
    if args.max_train is not None and args.max_train < len(X_tr_raw):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_tr_raw), size=args.max_train, replace=False)
        X_tr_raw, y_tr = X_tr_raw[idx], y_tr[idx]
        print(f"  Subsampled train to {args.max_train:,} rows")

    print(f"  X_train: {X_tr_raw.shape}  X_val: {X_va_raw.shape}  X_test: {X_te_raw.shape}")
    print(f"  TTF cap = {cap}s;  "
          f"y_test: {100*(y_te < cap).mean():.1f}% approaching failure")
    approaching_mask = y_te < cap

    X_tr_seq, X_tr_aux = reshape_features(X_tr_raw, feature_names)
    X_va_seq, X_va_aux = reshape_features(X_va_raw, feature_names)
    X_te_seq, X_te_aux = reshape_features(X_te_raw, feature_names)

    # Standardize aux features (tree models don't need this; nets do)
    aux_scaler = StandardScaler().fit(X_tr_aux)
    X_tr_aux = aux_scaler.transform(X_tr_aux).astype(np.float32)
    X_va_aux = aux_scaler.transform(X_va_aux).astype(np.float32)
    X_te_aux = aux_scaler.transform(X_te_aux).astype(np.float32)

    # Standardize SNR sequence (fit on train mean/std of the scalar signal)
    seq_mean = X_tr_seq.mean()
    seq_std  = X_tr_seq.std() + 1e-8
    X_tr_seq = ((X_tr_seq - seq_mean) / seq_std).astype(np.float32)
    X_va_seq = ((X_va_seq - seq_mean) / seq_std).astype(np.float32)
    X_te_seq = ((X_te_seq - seq_mean) / seq_std).astype(np.float32)

    # Scale target by cap for numerical stability (predictions stay in [0, 1])
    y_tr_s = (y_tr / cap).astype(np.float32)
    y_va_s = (y_va / cap).astype(np.float32)
    y_te_s = (y_te / cap).astype(np.float32)

    def make_loader(seq, aux, y, shuffle):
        ds = TensorDataset(torch.from_numpy(seq),
                           torch.from_numpy(aux),
                           torch.from_numpy(y))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = make_loader(X_tr_seq, X_tr_aux, y_tr_s, shuffle=True)
    val_loader   = make_loader(X_va_seq, X_va_aux, y_va_s, shuffle=False)
    test_loader  = make_loader(X_te_seq, X_te_aux, y_te_s, shuffle=False)

    # -------- Per-class helper (uses train_pkl if provided) --------
    stats_df = None
    if args.train_pkl and Path(args.train_pkl).exists():
        df_full = pd.read_pickle(args.train_pkl)
        stats_df = df_full.groupby("trajectory_id").agg(
            label=("failure_type", "first")
        ).reset_index()

    # -------- Run each requested model --------
    all_rows = []
    per_class_tables = {}

    for model_name in args.models:
        print("\n" + "=" * 72)
        print(f"MODEL: {model_name.upper()}  ({args.n_seeds} seeds)")
        print("=" * 72)

        model_rows = []
        last_preds = None
        for s in range(args.n_seeds):
            torch.manual_seed(s)
            np.random.seed(s)

            if model_name == "lstm":
                model = LSTMPredictor(hidden=64, num_layers=2, aux_dim=4, dropout=0.1)
            else:  # cnn
                model = CNN1DPredictor(channels=(32, 64, 64), aux_dim=4, dropout=0.1)

            print(f"\n  seed {s} — training ...")
            t0 = time.time()
            model, best_val_scaled = train_one_model(
                model, train_loader, val_loader,
                epochs=args.epochs, lr=args.lr, device=device, patience=args.patience,
            )
            train_time = time.time() - t0

            # Predict on test
            y_pred_scaled = predict_model(model, test_loader, device)
            y_pred = y_pred_scaled * cap
            y_pred = np.clip(y_pred, 0, cap)

            m = evaluate_model(f"{model_name.upper()}_seed{s}", y_te, y_pred,
                               approaching_mask)
            m["train_time_s"] = round(train_time, 1)
            m["best_val_mae_s"] = round(best_val_scaled * cap, 2)
            print(f"    -> MAE={m['MAE']:6.2f}  approaching MAE="
                  f"{m.get('MAE_approaching', float('nan')):6.2f}  "
                  f"R2={m['R2']:.3f}  time={m['train_time_s']}s")
            model_rows.append(m)

            # Save one model per seed
            torch.save(model.state_dict(),
                       out_dir / "models" / f"{model_name}_seed{s}.pt")
            last_preds = y_pred

        # Aggregate for this model
        df_m = pd.DataFrame(model_rows)
        df_m.to_csv(out_dir / f"metrics_per_seed_{model_name}.csv", index=False)

        def agg(col):
            if col not in df_m.columns:
                return None
            v = df_m[col].values
            if len(v) > 1:
                sem = sps.sem(v); ci = sps.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=sem)
                return v.mean(), v.std(ddof=1), ci[0], ci[1]
            return v.mean(), 0.0, v.mean(), v.mean()

        summ = {"Model": model_name.upper(), "n_seeds": args.n_seeds}
        for metric in ["MAE", "RMSE", "R2", "MAE_approaching", "R2_approaching"]:
            a = agg(metric)
            if a is None:
                continue
            summ[f"{metric}_mean"] = a[0]
            summ[f"{metric}_std"]  = a[1]
            if metric == "MAE":
                summ["MAE_CI_lower"] = a[2]
                summ["MAE_CI_upper"] = a[3]
        all_rows.append(summ)

        if stats_df is not None and last_preds is not None:
            pc = per_class_metrics(y_te, last_preds, traj_te, stats_df, cap)
            label_map = {0: "No failure", 1: "ECL", 2: "EDFA", 3: "NLI"}
            pc["label_name"] = pc["label"].map(label_map)
            pc.to_csv(out_dir / f"per_class_metrics_{model_name}.csv", index=False)
            per_class_tables[model_name] = pc

    # -------- Final summary --------
    pd.DataFrame(all_rows).to_csv(out_dir / "summary.csv", index=False)
    print("\n" + "=" * 72)
    print("DEEP-BASELINE SUMMARY")
    print("=" * 72)
    with pd.option_context("display.max_columns", None,
                           "display.width", 160,
                           "display.precision", 3):
        print(pd.DataFrame(all_rows).to_string(index=False))

    if per_class_tables:
        print("\nPer-class breakdown:")
        for name, pc in per_class_tables.items():
            print(f"\n[{name.upper()}]")
            print(pc.to_string(index=False))

    print(f"\nAll outputs in: {out_dir}/")
    print(">>> SEND BACK: summary.csv AND per_class_metrics_*.csv AND console log <<<")


if __name__ == "__main__":
    main()