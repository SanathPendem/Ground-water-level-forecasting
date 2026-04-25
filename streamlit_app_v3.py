# streamlit_app_v2.py
# ------------------------------------------------------------
# Groundwater Level Forecast UI (PCA + DWT + LSTM) — Streamlit
#
# Changes vs v1:
# ✅ No pre-filled defaults (starts blank-ish)
# ✅ Removed "Advanced" UI (history handled internally)
# ✅ Robust NaN/Inf guards + clipping to prevent unstable inference
# ✅ If prediction becomes NaN, shows diagnostics instead of silently continuing
# ------------------------------------------------------------

import os
import pickle
import numpy as np
import streamlit as st

import pywt
import torch
import torch.nn as nn


# =========================
# CONFIG — EDIT THIS PATH
# =========================
MODEL_PKL = r"C:\Users\akvsk\Downloads\Khyathi\Project\pca_dwt_lstm_bundle.pkl"


# =========================
# MODEL (same shape as training)
# =========================
class LSTMReg(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def dwt_denoise(x, wavelet: str, level: int):
    x = np.ascontiguousarray(np.array(x, dtype=np.float64, copy=True))
    # If history is constant/short, keep safe
    if len(x) < 4 or np.allclose(x, x[0]):
        return x.astype(np.float32)

    coeffs = pywt.wavedec(x, wavelet, level=level, mode="periodization")
    sigma = (np.median(np.abs(coeffs[-1])) / 0.6745) if len(coeffs[-1]) else 0.0
    uth = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0

    # If uth becomes 0 (or NaN), thresholding can create NaNs (division by zero)
    if (not np.isfinite(uth)) or uth == 0.0:
        return x.astype(np.float32)

    new = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(new, wavelet, mode="periodization")
    rec = rec[: len(x)]
    rec = np.nan_to_num(rec, nan=float(np.nanmean(x)))
    return np.array(rec, dtype=np.float32)


@st.cache_resource(show_spinner=False)
def load_bundle():
    if not os.path.exists(MODEL_PKL):
        raise FileNotFoundError(
            f"Model bundle not found at:\n{MODEL_PKL}\n\n"
            "Fix: edit MODEL_PKL in streamlit_app_v2.py to your actual .pkl path."
        )

    with open(MODEL_PKL, "rb") as f:
        bundle = pickle.load(f)

    scaler = bundle["scaler"]
    pca = bundle["pca"]
    meta = bundle.get("pipeline_meta", {})
    params = bundle["model_params"]

    lookback = int(params.get("lookback", 8))
    input_dim = int(params["input_dim"])
    hidden = int(params.get("hidden", 32))

    model = LSTMReg(input_dim=input_dim, hidden=hidden)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    exo_cols = meta.get(
        "exo_cols",
        ["rain_mm", "temp_c", "rh_pct", "evap_mm", "pumping_idx", "recharge_idx", "lat", "lon"],
    )
    dwt_wavelet = meta.get("dwt_wavelet", "db4")
    dwt_level = int(meta.get("dwt_level", 2))

    # A better default than 0.0 if your training stored it
    seed = float(meta.get("gwl_seed", 0.0))
    # If you stored training mean/std, you can use mean as seed too
    seed = float(meta.get("gwl_mean", seed))

    n_pc = int(getattr(pca, "n_components_", None) or pca.components_.shape[0])

    return scaler, pca, model, exo_cols, lookback, n_pc, dwt_wavelet, dwt_level, seed


def init_history(lookback: int, seed: float):
    if "gwl_hist" not in st.session_state or len(st.session_state["gwl_hist"]) != lookback:
        st.session_state["gwl_hist"] = [float(seed)] * lookback


def safe_clip(arr, lo, hi):
    arr = np.array(arr, dtype=np.float32, copy=True)
    arr = np.nan_to_num(arr, nan=0.0, posinf=hi, neginf=lo)
    return np.clip(arr, lo, hi)


def build_sequence(exo_vec, scaler, pca, gwl_hist, lookback, n_pc, dwt_wavelet, dwt_level):
    exo_vec = np.array(exo_vec, dtype=np.float32).reshape(1, -1)
    if not np.isfinite(exo_vec).all():
        raise ValueError("Inputs contain NaN/Inf. Please enter valid numeric values.")

    # Transform
    Xs = scaler.transform(exo_vec)

    # Guardrail: huge z-scores can destabilize LSTM → clip
    Xs = safe_clip(Xs, -6.0, 6.0)

    pcs = pca.transform(Xs).astype(np.float32)

    # Guardrail: clip PCs too (keeps inference stable)
    pcs = safe_clip(pcs, -10.0, 10.0)

    # Rolling GWL history
    hist = np.array(gwl_hist, dtype=np.float32)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)

    gwl_dn = dwt_denoise(hist, dwt_wavelet, dwt_level)

    # Final feature sequence
    seq = np.zeros((lookback, n_pc + 1), dtype=np.float32)
    seq[:, :n_pc] = pcs  # repeat current PCs across lookback
    seq[:, -1] = gwl_dn  # history across lookback

    if not np.isfinite(seq).all():
        raise ValueError("Internal features became NaN/Inf. Check input ranges or scaler/pca compatibility.")

    return seq.reshape(1, lookback, n_pc + 1), Xs, pcs, gwl_dn


def analysis_text(inp, pred):
    rain, evap = inp["rain_mm"], inp["evap_mm"]
    pump, rech = inp["pumping_idx"], inp["recharge_idx"]
    wb = rain - evap
    stress = pump - rech
    lines = [
        f"- Water-balance proxy (rain − evap): `{wb:.2f}`",
        f"- Stress proxy (pumping − recharge): `{stress:.2f}`",
        "- ✅ Recharge-supportive signal" if wb >= 0 else "- ⚠️ Recharge-weaker signal",
        "- ⚠️ Depletion-risk signal" if stress > 0 else "- ✅ Recovery-friendly signal" if stress < 0 else "- ℹ️ Balanced pumping/recharge",
        f"- Predicted GWL: `{pred:.3f}`",
    ]
    return lines


# =========================
# UI
# =========================
st.set_page_config(page_title="GWL Forecast", page_icon="💧", layout="centered")
st.title("💧 Groundwater Level (GWL) Forecast")
st.caption("Uses the saved **.pkl** bundle in code (no dataset, no upload).")

try:
    scaler, pca, model, exo_cols, lookback, n_pc, dwt_wavelet, dwt_level, seed = load_bundle()
except Exception as e:
    st.error(str(e))
    st.stop()

init_history(lookback, seed)

st.subheader("Enter inputs")

# No “mystery defaults”: leave fields at 0-ish, user fills them
c1, c2 = st.columns(2)
with c1:
    rain_mm = st.number_input("rain_mm", min_value=0.0, value=0.0, step=1.0, format="%.3f")
    temp_c = st.number_input("temp_c", min_value=-5.0, max_value=55.0, value=0.0, step=0.5, format="%.3f")
    rh_pct = st.number_input("rh_pct", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.3f")
    evap_mm = st.number_input("evap_mm", min_value=0.0, value=0.0, step=1.0, format="%.3f")
with c2:
    pumping_idx = st.number_input("pumping_idx", min_value=0.0, value=0.0, step=0.05, format="%.3f")
    recharge_idx = st.number_input("recharge_idx", min_value=0.0, value=0.0, step=0.05, format="%.3f")
    lat = st.number_input("lat", min_value=-90.0, max_value=90.0, value=0.0, step=0.001, format="%.6f")
    lon = st.number_input("lon", min_value=-180.0, max_value=180.0, value=0.0, step=0.001, format="%.6f")

inp = {
    "rain_mm": float(rain_mm),
    "temp_c": float(temp_c),
    "rh_pct": float(rh_pct),
    "evap_mm": float(evap_mm),
    "pumping_idx": float(pumping_idx),
    "recharge_idx": float(recharge_idx),
    "lat": float(lat),
    "lon": float(lon),
}

if st.button("Predict GWL", type="primary"):
    try:
        exo_vec = [inp[c] for c in exo_cols]
        X_seq, Xs, pcs, gwl_dn = build_sequence(
            exo_vec, scaler, pca, st.session_state["gwl_hist"], lookback, n_pc, dwt_wavelet, dwt_level
        )

        with torch.no_grad():
            xt = torch.from_numpy(X_seq).float()
            pred = float(model(xt).cpu().numpy().ravel()[0])

        if not np.isfinite(pred):
            st.error("Prediction became NaN/Inf. Showing diagnostics below.")
            st.write("**Diagnostics:**")
            st.write("Standardized X (after clip):", Xs)
            st.write("PCA PCs (after clip):", pcs)
            st.write("DWT GWL history:", gwl_dn)
            st.stop()

        st.session_state["gwl_hist"] = st.session_state["gwl_hist"][1:] + [pred]

        st.success("Prediction completed.")
        st.metric("Predicted GWL", f"{pred:.3f}")

        st.markdown("### Analysis")
        for line in analysis_text(inp, pred):
            st.write(line)

        st.markdown("### Rolling context (internal)")
        st.line_chart(st.session_state["gwl_hist"])

    except Exception as e:
        st.error(f"Prediction failed: {e}")
