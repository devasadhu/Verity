# Verity — Real-Time Transaction Intelligence Engine

A production-grade fraud detection system built from scratch — every component implemented without high-level ML libraries (no sklearn, no PyTorch, no Kafka, no RocksDB) to demonstrate deep understanding of how fintech systems work at the systems and ML level.

**131 tests passing. All four weeks complete.**

---

## Architecture

```
Transaction
    │
    ▼
┌─────────────────┐
│  Ingestion       │  UPI simulator · IEEE-CIS adapter · Message queue
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Storage Engine  │  C hash map · WAL · LSM tree · B-tree + inverted index
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Stream Processing│  Sliding windows · Count-Min Sketch · Rule engine (7 rules)
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│   ML Scoring    │  Focal loss · MLP · GBT (XGBoost-style) · Online learning · Ensemble
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│ Explainability  │  KernelSHAP · Model registry · PSI drift detector
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│   FastAPI        │  /score · /explain · /audit · shadow mode · fail strategies
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│  Alert Engine   │  Deduplication · Webhook delivery · Exponential backoff
└────────┬────────┘
         │
    ▼
┌─────────────────┐
│   Dashboard     │  Streamlit · Live feed · SHAP bar chart · PSI heatmap · Review queue
└─────────────────┘
```

---

## Performance

| Metric | Value |
|---|---|
| Scoring latency p50 | **0.013 ms** |
| Scoring latency p99 | **0.049 ms** |
| Throughput (single thread) | **67,771 txn/s** |
| SHAP explanation p50 | **1.2 ms** |
| SHAP explanation p99 | **2.2 ms** |
| PSI drift check (500-txn window, 23 features) | **2.8 ms** |

> Run `python benchmarks/run.py` to reproduce with trained models loaded.

---

## Repository Structure

```
verity/
├── ingestion/
│   ├── simulator.py          UPI transaction simulator (4 fraud types)
│   ├── ieee_adapter.py       IEEE-CIS 590K row adapter
│   └── message_queue.py      Append-only queue with CRC32, fsync
├── storage/
│   ├── hashmap.c / .h / .py  Open-addressing hash table; C hot path, Python fallback
│   ├── wal.py                Write-ahead log with crash recovery
│   ├── lsm.py                Two-level LSM tree (Memtable → SSTable → compaction)
│   ├── indexes.py            B-tree range index + inverted index
│   └── Makefile
├── stream/
│   ├── sketch.py             Count-Min Sketch (SHA-256 hash family)
│   ├── windows.py            Sliding windows 1m/5m/1h/24h, 17-feature vector
│   └── rules.py              7-rule engine with structured result
├── ml/
│   ├── losses.py             Focal loss with analytical gradient
│   ├── mlp.py                3-layer MLP: He init, BN, ReLU, dropout, Adam
│   ├── gbt.py                GBT with XGBoost-style second-order gradients
│   ├── online.py             Online learner with reservoir replay + drift adaptation
│   └── ensemble.py           Feature store + 0.4×MLP + 0.6×GBT ensemble
├── explainability/
│   ├── shap.py               KernelSHAP from scratch (Lundberg & Lee 2017)
│   ├── registry.py           Versioned model registry + immutable audit log
│   └── drift.py              PSI drift detector with save/load
├── api/
│   ├── main.py               FastAPI: /score /explain /audit /health /metrics
│   └── alerts.py             Alert engine: dedup cache + webhook + backoff
├── adversarial/
│   └── simulator.py          6 attack patterns + AttackSuite benchmark runner
├── dashboard/
│   └── app.py                Streamlit: live feed, score dist, PSI heatmap, review queue
├── benchmarks/
│   └── run.py                Latency, throughput, adversarial, SHAP, PSI benchmarks
├── tests/                    131 tests across all modules
├── scripts/
│   ├── download_data.py
│   ├── build.py
│   └── smoke_test.py
└── requirements.txt
```

---

## Setup

```bash
# Clone
git clone https://github.com/devasadhu/Verity && cd Verity

# Windows (Python 3.13)
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# Linux / HPC (builds C hash map)
python scripts/build.py

# Optional: download IEEE-CIS dataset (requires Kaggle API key)
python scripts/download_data.py

# Run tests
pytest tests\ -v

# Run smoke test
python scripts\smoke_test.py

# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start dashboard
streamlit run dashboard/app.py
```

---

## Components — Design Decisions

### Why C for the hash map?
The hot path is C, the application layer is Python — exactly how Redis is built. The C and Python implementations share an identical interface; the C library loads automatically on Linux/HPC, Python fallback activates on Windows where MinGW's 32-bit gcc can't link against Python 3.13's 64-bit runtime.

### Why LSM tree for writes?
Fraud workloads are write-heavy. LSM turns random writes into sequential writes by buffering in the Memtable and flushing sorted runs to disk. Sequential I/O is 10-100× faster than random I/O. This is why RocksDB is the default storage engine at Razorpay, Stripe, and most high-throughput payment processors.

### Why focal loss instead of cross-entropy?
Fraud data is 0.1-1% positive class. Cross-entropy lets a model predict "legit" for everything and achieve 99% accuracy — it learns nothing. Focal loss's focusing factor `(1-p_t)^γ` down-weights easy negatives so the model is forced to learn hard fraud cases. The IEEE-CIS dataset has ~286× class imbalance; focal loss is the correct tool.

### Why both MLP and GBT?
Neural nets capture feature interactions better. Tree models outperform on tabular data with clear decision boundaries. They make different types of errors. The ensemble averages uncorrelated errors and reduces variance. Weights: 0.4×MLP + 0.6×GBT (GBT weighted higher for tabular data).

### Why rules before ML?
Real fraud systems don't run ML on every transaction. Rules handle 80% of obvious cases in microseconds. ML fires only on ambiguous signals. This is how Razorpay, Paytm, and every production fraud system is architected. The rule engine short-circuits on the first BLOCK.

### Why KernelSHAP?
RBI guidelines on algorithmic financial decisions require automated blocks to be explainable. Every decision can be described in plain terms: "flagged because amount was 3.2× user average (+0.41), new device (+0.29), 2AM transaction (+0.18)." KernelSHAP is model-agnostic and works on the ensemble output — no white-box assumptions.

### Why shadow mode?
Fraud systems never go live directly. Running in shadow mode first — scoring without blocking, comparing to ground truth — is standard practice at every major payment processor. The operating mode can be hot-switched via `POST /mode/{mode}` without restart.

### Why PSI for drift detection?
Population Stability Index is the standard metric in credit risk model monitoring (Basel II/III). It's interpretable (`PSI < 0.10` stable, `≥ 0.20` retrain), computationally cheap (2.8ms for 23 features over 500 transactions), and maps directly to the regulatory framing used by Indian payment processors.

---

## API Reference

```
POST /score
  Body: { transaction_id, user_id, merchant_id, amount, mcc, city, device_id }
  Returns: { risk_score, rule_score, ml_score, decision, shadow_mode,
             model_version, latency_ms }

POST /explain
  Body: same as /score + feature_vector
  Returns: { risk_score, decision, baseline, top_features, explanation }

GET  /audit/{transaction_id}
  Returns: all audit log entries for the transaction

GET  /health
  Returns: { status, mode, model_loaded, uptime_s }

GET  /metrics
  Returns: { total_scored, decisions, p50_ms, p99_ms, drift_alert }

POST /mode/{mode}
  Modes: normal | shadow | degraded | fail_open | fail_closed
  Hot-switches operating mode without restart.
```

---

## Adversarial Test Suite

Six attack patterns validate detection across all layers:

| Attack | Evasion Strategy | Primary Counter |
|---|---|---|
| VelocityAttack | Burst just below 1-min rule threshold | ML txn_count z-score |
| DeviceFarm | Many users per device | Count-Min Sketch frequency |
| LowAndSlow | Gradual baseline poisoning over days | 24h window + ML |
| AmountSplitting | Split target amount across merchants | 24h amount velocity |
| GeoSpoofing | Impossible travel > 61-min intervals | unique_cities_24h feature |
| BotGenerated | Machine-perfect 60s timing | Timing regularity feature |

Run: `python -c "from adversarial.simulator import AttackSuite; ..."`

---

## Test Coverage

| Module | Tests |
|---|---|
| test_simulator.py | 4 |
| test_message_queue.py | 4 |
| test_storage.py | 12 |
| test_lsm.py | 11 |
| test_indexes.py | 11 |
| test_stream.py | 14 |
| test_ml.py | 26 |
| test_week4.py (shap, registry, drift, alerts, adversarial) | 44 |
| **Total** | **131** |

---

## Regulatory Framing (RBI Compliance)

| Requirement | Implementation |
|---|---|
| Explainability | KernelSHAP per-decision explanation |
| Audit trail | Immutable append-only JSONL audit log with fsync |
| Model versioning | Content-hash versioned model registry |
| Drift monitoring | PSI per-feature with alert at PSI ≥ 0.20 |
| Human review | Review queue with feedback loop to online learner |
| Safe rollout | Shadow mode with decision logging before enforcement |

---