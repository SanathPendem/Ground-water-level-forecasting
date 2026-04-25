import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pywt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# =========================
# CONFIG
# =========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_CSV = r"C:\Users\akvsk\Downloads\Khyathi\Project\synthetic_gwl_india_quarterly_1991_2020.csv"

TRAIN_ON_ALL_STATES = True
LOOKBACK = 8
CUTOFF_DATE = pd.Timestamp("2016-01-01")

# Accuracy band stop
TARGET_R2_MIN = 0.92
TARGET_R2_MAX = 0.95

# DWT settings
DWT_WAVELET = "db4"
DWT_LEVEL = 2

# Output model bundle
MODEL_PKL = r"C:\Users\akvsk\Downloads\Khyathi\Project\pca_dwt_lstm_bundle.pkl"

# =========================
# HELPERS
# =========================
def dwt_denoise(x, wavelet=DWT_WAVELET, level=DWT_LEVEL):
    # Make sure PyWavelets gets a writeable, contiguous buffer
    x = np.ascontiguousarray(np.array(x, dtype=np.float64, copy=True))

    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if len(coeffs[-1]) else 0.0
    uth = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0
    new = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(new, wavelet, mode="periodization")
    return rec[:len(x)]


def build_sequences(df_station, feat_cols, lookback):
    vals = df_station[feat_cols].to_numpy(dtype=np.float32)
    y = df_station["gwl"].to_numpy(dtype=np.float32)
    dts = df_station["date"].to_numpy()
    Xs, ys, ds = [], [], []
    for t in range(lookback, len(df_station)):
        Xs.append(vals[t - lookback:t])
        ys.append(y[t])
        ds.append(dts[t])
    return np.stack(Xs), np.array(ys), np.array(ds)


class LSTMReg(nn.Module):
    def __init__(self, input_dim, hidden=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def metrics(y_true, y_pred, name=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# =========================
# LOAD
# =========================
use = pd.read_csv(DATA_CSV)
use["date"] = pd.to_datetime(use["date"])
use = use.sort_values(["station_id", "date"]).reset_index(drop=True)

if not TRAIN_ON_ALL_STATES:
    use = use[use["state"] == "West Bengal"].copy()

# =========================
# DWT ON TARGET
# =========================
use["gwl_denoised"] = 0.0
for sid, g in use.groupby("station_id", sort=False):
    use.loc[g.index, "gwl_denoised"] = dwt_denoise(g["gwl"].to_numpy(copy=True))

use["is_test"] = use["date"] >= CUTOFF_DATE

exo_cols = ["rain_mm", "temp_c", "rh_pct", "evap_mm", "pumping_idx", "recharge_idx", "lat", "lon"]

# =========================
# SCALER + PCA (FIT ONLY ON TRAIN)
# =========================
train_mask = ~use["is_test"]

scaler = StandardScaler()
X_train_exo = scaler.fit_transform(use.loc[train_mask, exo_cols])

pca = PCA(n_components=0.95, random_state=SEED)
pca.fit(X_train_exo)

X_all_pca = pca.transform(scaler.transform(use[exo_cols]))
n_pc = X_all_pca.shape[1]

for i in range(n_pc):
    use[f"PC{i + 1}"] = X_all_pca[:, i]

feat_cols = [f"PC{i + 1}" for i in range(n_pc)] + ["gwl_denoised"]

# =========================
# SEQUENCES
# =========================
X_list, y_list, dt_list, st_list = [], [], [], []

for sid, g in use.groupby("station_id", sort=False):
    Xs, ys, dts = build_sequences(g, feat_cols, LOOKBACK)
    X_list.append(Xs)
    y_list.append(ys)
    dt_list.append(dts)
    st_list.append(np.array([g["state"].iloc[0]] * len(ys)))

X = np.concatenate(X_list)
y = np.concatenate(y_list)
dts = pd.to_datetime(np.concatenate(dt_list))
sts = np.concatenate(st_list)

is_test = dts >= CUTOFF_DATE
X_train, X_test = X[~is_test], X[is_test]
y_train, y_test = y[~is_test], y[is_test]
st_test = sts[is_test]

# =========================
# TRAIN
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMReg(input_dim=X_train.shape[2], hidden=32).to(device)

opt = torch.optim.Adam(model.parameters(), lr=2e-3)
loss_fn = nn.MSELoss()

Xtr = torch.from_numpy(X_train).float()
ytr = torch.from_numpy(y_train).float().unsqueeze(1)
ds = TensorDataset(Xtr, ytr)

val_size = int(len(ds) * 0.1)
train_size = len(ds) - val_size

train_ds, val_ds = random_split(
    ds, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)


def val_r2():
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            ps.append(model(xb).cpu().numpy().ravel())
            ys.append(yb.numpy().ravel())
    return r2_score(np.concatenate(ys), np.concatenate(ps))


best_state = None
best_val = 1e18
pat = 0
PATIENCE = 6

hist_tr, hist_va, hist_r2 = [], [], []

for epoch in range(1, 40):
    model.train()
    tr = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tr += loss.item() * len(xb)
    tr /= train_size

    model.eval()
    va = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            va += loss_fn(model(xb), yb).item() * len(xb)
    va /= val_size

    r2v = val_r2()
    hist_tr.append(tr)
    hist_va.append(va)
    hist_r2.append(r2v)

    print(f"Epoch {epoch:02d} | train={tr:.4f} val={va:.4f} val_R2={r2v:.4f}")

    if va < best_val - 1e-5:
        best_val = va
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pat = 0
    else:
        pat += 1

    # keep accuracy inside requested band
    if TARGET_R2_MIN <= r2v <= TARGET_R2_MAX and epoch >= 6:
        print("Stop: reached target validation band.")
        break
    if pat >= PATIENCE:
        print("Stop: no improvement.")
        break

# restore best weights
model.load_state_dict(best_state)

# =========================
# SAVE BUNDLE (.pkl)
# =========================
bundle = {
    "scaler": scaler,                 # StandardScaler (fit on train)
    "pca": pca,                       # PCA (fit on train)
    "model_state_dict": best_state,   # frozen weights on CPU
    "model_params": {
        "input_dim": int(X_train.shape[2]),
        "hidden": int(model.hidden),
        "lookback": int(LOOKBACK),
    },
    "pipeline_meta": {
        "exo_cols": exo_cols,
        "feat_cols": feat_cols,
        "cutoff_date": str(CUTOFF_DATE.date()),
        "train_on_all_states": bool(TRAIN_ON_ALL_STATES),
        "dwt_wavelet": DWT_WAVELET,
        "dwt_level": int(DWT_LEVEL),
        "seed": int(SEED),
    },
    "versions": {
        "python": None,
        "torch": torch.__version__,
    }
}

with open(MODEL_PKL, "wb") as f:
    pickle.dump(bundle, f)

print(f"\nSaved model bundle to: {MODEL_PKL}")

# =========================
# PREDICT + METRICS
# =========================
Xt = torch.from_numpy(X_test).float().to(device)
model.eval()
with torch.no_grad():
    pred_test = model(Xt).cpu().numpy().ravel()

metrics(y_test, pred_test, "Test (All)")
wb_mask = (st_test == "West Bengal")
metrics(y_test[wb_mask], pred_test[wb_mask], "Test (West Bengal)")

# =========================
# PLOTS
# =========================
plt.figure(figsize=(8, 4))
plt.plot(hist_tr, label="train")
plt.plot(hist_va, label="val")
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(hist_r2)
plt.title("Validation R2")
plt.xlabel("Epoch")
plt.ylabel("R2")
plt.show()

y_true = y_test[wb_mask]
y_pred = pred_test[wb_mask]

plt.figure(figsize=(10, 4))
plt.plot(y_true[:300], label="Actual")
plt.plot(y_pred[:300], label="Predicted")
plt.title("WB test slice")
plt.legend()
plt.show()

plt.figure(figsize=(5, 5))
plt.scatter(y_true, y_pred, alpha=0.4)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("WB: Pred vs Actual")
plt.show()

plt.figure(figsize=(8, 4))
plt.hist(y_true - y_pred, bins=40)
plt.title("WB residuals")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.show()
