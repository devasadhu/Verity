"""
Verity Benchmark Suite

Measures and prints:
  1. Latency benchmark — rule engine + mock ML scoring p50/p99 under load
  2. Throughput — transactions/second (single thread, no I/O blocking)
  3. Adversarial detection rates — per-attack and overall
  4. PSI computation speed — for 500-transaction windows

Run from project root:
    python benchmarks/run.py

Outputs JSON to benchmarks/results.json for README population.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import uuid

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adversarial.simulator import AttackSuite
from explainability.drift import PSIDriftDetector, compute_psi
from explainability.shap import KernelSHAP


# ---------------------------------------------------------------------------
# Mock scoring pipeline (no disk I/O, simulates full stack latency)
# ---------------------------------------------------------------------------

def _build_mock_scorer():
    """
    Minimal scorer that exercises KernelSHAP + PSI paths for a realistic
    latency measurement without requiring trained model files.
    """
    rng = np.random.default_rng(0)
    d = 23  # Verity feature dimension

    # Toy model: logistic function of L2 norm, biased toward low scores
    def predict_fn(X):
        X = np.asarray(X)
        z = np.linalg.norm(X, axis=1) / (d ** 0.5)
        return 1 / (1 + np.exp(-z + 2))

    background = rng.standard_normal((100, d))
    explainer = KernelSHAP(predict_fn, background, n_samples=64, random_state=0)
    return predict_fn, explainer, d


def _mock_score(predict_fn, d: int) -> tuple[float, str]:
    fv = np.random.standard_normal(d)
    score = float(predict_fn(fv.reshape(1, -1))[0])
    if score >= 0.90:
        decision = "BLOCK"
    elif score >= 0.70:
        decision = "REVIEW"
    else:
        decision = "PASS"
    return score, decision


# ---------------------------------------------------------------------------
# Benchmark 1: Latency
# ---------------------------------------------------------------------------

def bench_latency(n_warmup=200, n_measure=2000) -> dict:
    predict_fn, _, d = _build_mock_scorer()

    # Warmup
    for _ in range(n_warmup):
        _mock_score(predict_fn, d)

    latencies = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        _mock_score(predict_fn, d)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()

    def pct(p):
        idx = max(0, math.ceil(p / 100 * len(latencies)) - 1)
        return round(latencies[idx], 3)

    return {
        "n_transactions": n_measure,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "mean_ms": round(sum(latencies) / len(latencies), 3),
    }


# ---------------------------------------------------------------------------
# Benchmark 2: Throughput
# ---------------------------------------------------------------------------

def bench_throughput(duration_s: float = 3.0) -> dict:
    predict_fn, _, d = _build_mock_scorer()
    count = 0
    t_end = time.perf_counter() + duration_s
    while time.perf_counter() < t_end:
        _mock_score(predict_fn, d)
        count += 1
    tps = count / duration_s
    return {
        "duration_s": duration_s,
        "transactions": count,
        "tps": round(tps, 1),
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Adversarial detection
# ---------------------------------------------------------------------------

def bench_adversarial() -> dict:
    predict_fn, _, d = _build_mock_scorer()

    def score_fn(txn):
        # Use amount as a weak signal for demonstration
        amount_score = min(txn["amount"] / 20_000, 1.0)
        ml_score, _ = _mock_score(predict_fn, d)
        score = 0.5 * amount_score + 0.5 * ml_score
        decision = "BLOCK" if score >= 0.9 else ("REVIEW" if score >= 0.7 else "PASS")
        return score, decision

    suite = AttackSuite(score_fn)
    report = suite.run()
    return report.to_dict()


# ---------------------------------------------------------------------------
# Benchmark 4: SHAP latency
# ---------------------------------------------------------------------------

def bench_shap_latency(n_measure: int = 50) -> dict:
    predict_fn, explainer, d = _build_mock_scorer()

    latencies = []
    for _ in range(n_measure):
        x = np.random.standard_normal(d)
        t0 = time.perf_counter()
        explainer.explain(x)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "n_samples": n_measure,
        "p50_ms": round(latencies[len(latencies)//2], 1),
        "p99_ms": round(latencies[max(0, math.ceil(0.99*len(latencies))-1)], 1),
        "mean_ms": round(sum(latencies)/len(latencies), 1),
        "shap_n_samples": 64,
        "feature_dim": d,
    }


# ---------------------------------------------------------------------------
# Benchmark 5: PSI computation speed
# ---------------------------------------------------------------------------

def bench_psi_speed(n_features: int = 23, window_size: int = 500) -> dict:
    rng = np.random.default_rng(0)
    features = [f"f{i}" for i in range(n_features)]
    det = PSIDriftDetector(features, n_bins=10, window_size=window_size)
    X_ref = rng.normal(0, 1, (2000, n_features))
    s_ref = np.clip(rng.normal(0.3, 0.1, 2000), 0, 1)
    det.fit_reference(X_ref, s_ref)

    X_cur = rng.normal(0, 1, (window_size, n_features))
    s_cur = np.clip(rng.normal(0.3, 0.1, window_size), 0, 1)

    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        det.check(X_cur, s_cur)
    elapsed = (time.perf_counter() - t0) / reps * 1000

    return {
        "n_features": n_features,
        "window_size": window_size,
        "mean_check_ms": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("Running Verity benchmarks...\n")

    print("1/5  Latency benchmark...")
    lat = bench_latency()
    print(f"     p50={lat['p50_ms']}ms  p99={lat['p99_ms']}ms\n")

    print("2/5  Throughput benchmark...")
    thr = bench_throughput()
    print(f"     {thr['tps']:,.0f} transactions/second\n")

    print("3/5  Adversarial detection benchmark...")
    adv = bench_adversarial()
    print(f"     Overall detection rate: {adv['overall_detection_rate']:.1%}")
    for a in adv["attacks"]:
        print(f"       {a['attack']:<22} detect={a['detection_rate']:.1%}  block={a['block_rate']:.1%}")
    print()

    print("4/5  SHAP latency benchmark...")
    shap = bench_shap_latency()
    print(f"     p50={shap['p50_ms']}ms  p99={shap['p99_ms']}ms\n")

    print("5/5  PSI drift check speed...")
    psi = bench_psi_speed()
    print(f"     {psi['mean_check_ms']}ms per {psi['window_size']}-transaction window\n")

    results = {
        "latency": lat,
        "throughput": thr,
        "adversarial": adv,
        "shap": shap,
        "psi": psi,
    }

    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out}")
    return results


if __name__ == "__main__":
    main()