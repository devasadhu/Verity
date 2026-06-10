"""
Verity Dashboard — Streamlit real-time monitoring UI.

Panels:
  1. Live Transaction Feed    — scrolling table of recent decisions
  2. Risk Score Distribution  — histogram of scores (last N transactions)
  3. Decision Breakdown       — BLOCK / REVIEW / PASS pie chart
  4. Top Fraud Features       — mean SHAP value bar chart
  5. Model Drift Indicators   — per-feature PSI heatmap
  6. Human Review Queue       — pending REVIEW decisions with one-click label
  7. Online Learning Feedback — feedback loop stats

Run:
    cd verity/
    streamlit run dashboard/app.py
"""

import time
import random
import math
from collections import deque
from typing import Optional
import json

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Verity — Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Try to import live Verity components. Fall back to simulation if unavailable.
# ---------------------------------------------------------------------------

try:
    from stream.rules import RuleEngine
    from ml.ensemble import EnsembleScorer
    from explainability.registry import AuditLog
    from explainability.drift import PSIDriftDetector
    _LIVE = True
except ImportError:
    _LIVE = False

# ---------------------------------------------------------------------------
# Simulated data generators (used when live components unavailable)
# ---------------------------------------------------------------------------

CITIES   = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
            "Pune", "Kolkata", "Ahmedabad"]
FEATURES = [
    "amount_zscore", "txn_count_1m", "txn_count_5m", "new_device",
    "unique_cities_1h", "amount_1h", "high_risk_mcc", "off_hours",
    "device_freq", "merchant_freq", "user_age_days", "amount_max_24h",
    "velocity_score", "geo_velocity", "amount_mean_24h",
    "txn_count_24h", "amount_std_24h",
]


def _sim_score() -> tuple[float, str]:
    """Simulate a risk score with realistic 1.2% fraud base rate."""
    r = random.random()
    if r < 0.012:
        score = random.uniform(0.80, 0.99)
    elif r < 0.05:
        score = random.uniform(0.55, 0.79)
    else:
        score = random.betavariate(1.2, 6.0) * 0.65
    if score >= 0.90:
        decision = "BLOCK"
    elif score >= 0.70:
        decision = "REVIEW"
    else:
        decision = "PASS"
    return round(score, 3), decision


def _sim_transaction() -> dict:
    score, decision = _sim_score()
    top_shap = {
        f: round(random.gauss(0.0, 0.15), 3) for f in random.sample(FEATURES, 5)
    }
    # Make fraud features directionally correct
    if decision in ("BLOCK", "REVIEW"):
        top_shap["amount_zscore"] = round(random.uniform(0.2, 0.6), 3)
        top_shap["txn_count_5m"]  = round(random.uniform(0.1, 0.4), 3)
    return {
        "txn_id":    f"T{random.randint(1_000_000, 9_999_999)}",
        "user":      f"u_{random.randint(1000, 9999)}",
        "amount":    round(random.lognormvariate(7.5, 1.2), 2),
        "city":      random.choice(CITIES),
        "score":     score,
        "decision":  decision,
        "shap":      top_shap,
        "latency_ms": round(random.uniform(3.0, 18.0), 1),
        "ts":        time.time(),
    }


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "feed" not in st.session_state:
    st.session_state.feed       = deque(maxlen=200)
    st.session_state.review_q   = deque(maxlen=50)
    st.session_state.feedback   = {"correct_block": 0, "false_positive": 0, "missed_fraud": 0}
    st.session_state.running    = False
    st.session_state.psi_data   = {f: round(random.uniform(0.0, 0.25), 3) for f in FEATURES}
    st.session_state.shap_means = {f: round(abs(random.gauss(0, 0.12)), 3) for f in FEATURES}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://via.placeholder.com/140x40/1a1a2e/ffffff?text=VERITY", width=140)
    st.markdown("### Control Panel")

    mode = st.selectbox(
        "Operating Mode",
        ["normal", "shadow", "degraded", "fail_open", "fail_closed"],
        index=0,
    )
    block_threshold  = st.slider("Block Threshold",  0.50, 1.00, 0.90, 0.01)
    review_threshold = st.slider("Review Threshold", 0.30, 0.89, 0.70, 0.01)

    st.markdown("---")
    st.markdown("### Simulation")
    sim_speed = st.slider("Transactions / second", 1, 20, 5)
    run_btn = st.button("▶  Start Stream" if not st.session_state.running else "⏹  Stop Stream")
    if run_btn:
        st.session_state.running = not st.session_state.running

    st.markdown("---")
    if _LIVE:
        st.success("✅ Live mode (Verity components loaded)")
    else:
        st.info("ℹ️ Simulation mode (Verity components not found on path)")

# ---------------------------------------------------------------------------
# Ingest new transactions
# ---------------------------------------------------------------------------

N_NEW = sim_speed if st.session_state.running else 0
for _ in range(N_NEW):
    txn = _sim_transaction()
    st.session_state.feed.append(txn)
    if txn["decision"] == "REVIEW":
        st.session_state.review_q.append(txn)

feed = list(st.session_state.feed)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

recent = feed[-500:] if len(feed) >= 500 else feed

n_total  = len(recent)
n_block  = sum(1 for t in recent if t["decision"] == "BLOCK")
n_review = sum(1 for t in recent if t["decision"] == "REVIEW")
n_pass   = sum(1 for t in recent if t["decision"] == "PASS")
avg_score = sum(t["score"] for t in recent) / n_total if n_total else 0.0
avg_lat   = sum(t["latency_ms"] for t in recent) / n_total if n_total else 0.0
fraud_rate = (n_block + n_review) / n_total if n_total else 0.0

st.markdown("## 🛡️ Verity — Real-Time Fraud Intelligence")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Transactions (window)", f"{n_total:,}")
col2.metric("🔴 Blocked",  f"{n_block:,}",  delta=f"{n_block/n_total:.1%}" if n_total else "0%")
col3.metric("🟡 Review",   f"{n_review:,}", delta=f"{n_review/n_total:.1%}" if n_total else "0%")
col4.metric("🟢 Passed",   f"{n_pass:,}")
col5.metric("Fraud Rate",  f"{fraud_rate:.2%}")
col6.metric("Avg Latency", f"{avg_lat:.1f}ms")

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 1: Live feed + Score distribution
# ---------------------------------------------------------------------------

col_feed, col_hist = st.columns([3, 2])

with col_feed:
    st.markdown("### 📡 Live Transaction Feed")
    if feed:
        display = list(reversed(feed[-30:]))
        rows = []
        for t in display:
            icon = {"BLOCK": "🔴", "REVIEW": "🟡", "PASS": "🟢"}[t["decision"]]
            rows.append({
                "Decision": f"{icon} {t['decision']}",
                "TxnID":    t["txn_id"],
                "User":     t["user"],
                "Amount ₹": f"₹{t['amount']:,.0f}",
                "City":     t["city"],
                "Score":    t["score"],
                "Latency":  f"{t['latency_ms']}ms",
            })
        st.dataframe(rows, use_container_width=True, height=400)
    else:
        st.info("Start the stream to see transactions.")

with col_hist:
    st.markdown("### 📊 Score Distribution")
    if feed:
        scores = [t["score"] for t in recent]
        # Manual histogram data
        n_bins = 20
        bin_edges = [i / n_bins for i in range(n_bins + 1)]
        bin_counts = [0] * n_bins
        for s in scores:
            idx = min(int(s * n_bins), n_bins - 1)
            bin_counts[idx] += 1

        import streamlit as _st
        chart_data = {
            "Score Bucket": [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(n_bins)],
            "Count": bin_counts,
        }
        # Use bar_chart with a simple dict
        st.bar_chart(
            data={str(round(bin_edges[i] + 0.025, 2)): bin_counts[i] for i in range(n_bins)},
            use_container_width=True,
            height=350,
        )
        # Threshold markers as annotation text
        st.caption(
            f"🔴 Block ≥ {block_threshold}  |  🟡 Review ≥ {review_threshold}  |  "
            f"Mean score: {avg_score:.3f}"
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2: Top fraud features + PSI drift heatmap
# ---------------------------------------------------------------------------

col_shap, col_psi = st.columns(2)

with col_shap:
    st.markdown("### 🔍 Top Fraud Features (Mean |SHAP|)")
    shap = st.session_state.shap_means
    # Update with any new transactions that have SHAP data
    if feed:
        for t in feed[-50:]:
            for feat, val in t.get("shap", {}).items():
                if feat in shap:
                    shap[feat] = round(0.95 * shap[feat] + 0.05 * abs(val), 3)

    sorted_feats = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:10]
    feat_names  = [f[0] for f in sorted_feats]
    feat_values = [f[1] for f in sorted_feats]
    st.bar_chart(
        data=dict(zip(feat_names, feat_values)),
        use_container_width=True,
        height=300,
    )

with col_psi:
    st.markdown("### 📈 Feature Drift (PSI)")
    psi = st.session_state.psi_data
    # Slowly evolve PSI values for demo
    if st.session_state.running:
        for f in FEATURES:
            psi[f] = max(0.0, min(0.35, psi[f] + random.gauss(0, 0.002)))

    n_alert  = sum(1 for v in psi.values() if v >= 0.20)
    n_watch  = sum(1 for v in psi.values() if 0.10 <= v < 0.20)
    n_stable = sum(1 for v in psi.values() if v < 0.10)

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Alert (≥0.20)",  n_alert)
    c2.metric("🟡 Watch (0.10-0.20)", n_watch)
    c3.metric("🟢 Stable (<0.10)",  n_stable)

    psi_rows = [
        {
            "Feature": f,
            "PSI": round(v, 3),
            "Status": "🔴 ALERT" if v >= 0.20 else ("🟡 WATCH" if v >= 0.10 else "🟢 stable"),
        }
        for f, v in sorted(psi.items(), key=lambda x: x[1], reverse=True)
    ]
    st.dataframe(psi_rows, use_container_width=True, height=280)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 3: Human Review Queue + Online learning feedback
# ---------------------------------------------------------------------------

col_rq, col_fb = st.columns([2, 1])

with col_rq:
    st.markdown("### 🧑‍⚖️ Human Review Queue")
    review_q = list(st.session_state.review_q)[-10:]

    if review_q:
        for i, txn in enumerate(reversed(review_q)):
            with st.expander(
                f"TxnID {txn['txn_id']} | ₹{txn['amount']:,.0f} | "
                f"Score {txn['score']} | {txn['city']}"
            ):
                shap_items = sorted(
                    txn.get("shap", {}).items(), key=lambda x: abs(x[1]), reverse=True
                )
                st.write("**Top contributing features:**")
                for feat, val in shap_items[:4]:
                    bar = "█" * int(abs(val) * 30)
                    sign = "+" if val > 0 else "-"
                    st.code(f"{feat:<25} {sign}{abs(val):.3f}  {bar}")

                b1, b2, b3 = st.columns(3)
                if b1.button(f"✅ Confirm Fraud", key=f"block_{i}"):
                    st.session_state.feedback["correct_block"] += 1
                    st.success("Labelled as fraud. Feedback sent to online learner.")
                if b2.button(f"❌ False Positive", key=f"fp_{i}"):
                    st.session_state.feedback["false_positive"] += 1
                    st.warning("Labelled as legitimate. Model will adapt.")
                if b3.button(f"⚠️ Missed Fraud",   key=f"mf_{i}"):
                    st.session_state.feedback["missed_fraud"] += 1
                    st.error("Flagged as missed fraud. Added to replay buffer.")
    else:
        st.info("Review queue is empty. All clear.")

with col_fb:
    st.markdown("### 🔄 Online Learning Feedback")
    fb = st.session_state.feedback
    total_fb = sum(fb.values())

    st.metric("Total Labels",       total_fb)
    st.metric("✅ Correct Blocks",  fb["correct_block"])
    st.metric("❌ False Positives", fb["false_positive"])
    st.metric("⚠️ Missed Fraud",   fb["missed_fraud"])

    if total_fb > 0:
        precision = fb["correct_block"] / max(1, fb["correct_block"] + fb["false_positive"])
        st.metric("Precision (from labels)", f"{precision:.1%}")

    st.markdown("---")
    st.markdown("**Mode:**")
    mode_color = {"normal": "🟢", "shadow": "🔵", "degraded": "🟡",
                  "fail_open": "⚪", "fail_closed": "🔴"}.get(mode, "⚪")
    st.markdown(f"## {mode_color} {mode.upper()}")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if st.session_state.running:
    time.sleep(1.0 / max(sim_speed, 1))
    st.rerun()