"""
Adversarial Simulator — six attack patterns for stress-testing Verity.

Each attack is a generator that yields synthetic transactions crafted to
evade one or more of the detection layers (rules, ML, or both).

Attack patterns:
    1. VelocityAttack      — burst of transactions just below rule thresholds
    2. DeviceFarm          — many user IDs sharing a small pool of devices
    3. LowAndSlow          — gradual escalation over hours/days
    4. AmountSplitting     — large amounts split into smaller transactions
    5. GeoSpoofing         — impossible geo-velocity without triggering rule
    6. BotGenerated        — perfectly regular timing (non-human)

Each generator yields dicts in Verity's internal transaction schema.
An `AttackSuite` convenience class runs all attacks and scores each via a
provided scoring function, then returns a structured benchmark report.
"""

from __future__ import annotations

import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Generator, Optional


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _txn(
    user_id: str,
    merchant_id: str,
    amount: float,
    mcc: str,
    city: str,
    device_id: str,
    timestamp: Optional[float] = None,
    metadata: Optional[dict] = None,
) -> dict:
    return {
        "transaction_id": str(uuid.uuid4()),
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": round(amount, 2),
        "mcc": mcc,
        "city": city,
        "device_id": device_id,
        "timestamp": timestamp or time.time(),
        "metadata": metadata or {},
    }


CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
          "Pune", "Kolkata", "Ahmedabad", "Jaipur", "Surat"]
MCCS = ["5411", "5912", "4812", "5999", "7011", "5734", "5045"]
HIGH_RISK_MCCS = ["6051", "7995", "5933"]   # crypto, gambling, pawn shops


# ---------------------------------------------------------------------------
# Attack 1: Velocity Attack
# ---------------------------------------------------------------------------

def velocity_attack(
    n_txns: int = 50,
    amount: float = 4500.0,
    interval_s: float = 30.0,
    user_id: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Burst of transactions with inter-arrival time just above the 1-minute
    rule threshold (default: 3 txns/min triggers the rule).

    Strategy: send at ~2.9 transactions/minute to stay below the rule
    while still transacting at a high rate.

    Evasion insight: rules fire on count ≥ threshold; attacker stays at
    threshold - 1. ML should catch this via z-score on transaction frequency.
    """
    uid = user_id or f"u_vel_{uuid.uuid4().hex[:6]}"
    device = f"dev_{uuid.uuid4().hex[:8]}"
    merchant = f"merch_{uuid.uuid4().hex[:6]}"
    t = time.time()

    for i in range(n_txns):
        yield _txn(
            user_id=uid,
            merchant_id=merchant,
            amount=amount + random.uniform(-50, 50),
            mcc=random.choice(MCCS),
            city="Mumbai",
            device_id=device,
            timestamp=t + i * interval_s,
            metadata={"attack": "velocity", "sequence": i},
        )


# ---------------------------------------------------------------------------
# Attack 2: Device Farm
# ---------------------------------------------------------------------------

def device_farm_attack(
    n_users: int = 200,
    n_devices: int = 5,
    txns_per_user: int = 3,
    amount_range: tuple = (500.0, 2000.0),
) -> Generator[dict, None, None]:
    """
    Many user IDs sharing a small pool of devices — classic bot farm pattern.

    Each device is used by n_users / n_devices different users.
    Rule engine's device-farm rule fires on count-min sketch frequency:
    devices appearing for > K distinct users.

    Evasion attempt: spread transactions across time to reduce sketch density.
    """
    devices = [f"dev_farm_{i:03d}" for i in range(n_devices)]
    t = time.time()

    for u_idx in range(n_users):
        uid = f"u_farm_{u_idx:04d}"
        device = devices[u_idx % n_devices]
        for j in range(txns_per_user):
            yield _txn(
                user_id=uid,
                merchant_id=f"merch_{random.randint(1, 20):03d}",
                amount=random.uniform(*amount_range),
                mcc=random.choice(MCCS),
                city=random.choice(CITIES),
                device_id=device,
                timestamp=t + (u_idx * txns_per_user + j) * 20,
                metadata={"attack": "device_farm", "device_pool_size": n_devices},
            )


# ---------------------------------------------------------------------------
# Attack 3: Low-and-Slow
# ---------------------------------------------------------------------------

def low_and_slow_attack(
    n_days: int = 7,
    final_amount: float = 50_000.0,
    user_id: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Gradual amount escalation over n_days.

    Day 1: small amounts (500-2000 INR) to establish a normal baseline.
    Days 2-5: moderate escalation.
    Days 6-7: large transactions that would normally trigger z-score alert,
              but the recent window stats have been poisoned by the escalation.

    Evasion insight: window aggregator uses a rolling window; by slowly
    raising the baseline, the z-score of the final large transaction is
    artificially lowered.
    """
    uid = user_id or f"u_las_{uuid.uuid4().hex[:6]}"
    device = f"dev_{uuid.uuid4().hex[:8]}"
    t = time.time() - n_days * 86400

    txns_per_day = 4
    for day in range(n_days):
        fraction = day / (n_days - 1)  # 0.0 → 1.0
        # Exponential escalation
        day_amount = 500 + (final_amount - 500) * (fraction ** 2)
        for j in range(txns_per_day):
            jitter = random.uniform(-0.1, 0.1) * day_amount
            yield _txn(
                user_id=uid,
                merchant_id=f"merch_{random.randint(1, 10):03d}",
                amount=max(100, day_amount + jitter),
                mcc=random.choice(MCCS),
                city="Bangalore",
                device_id=device,
                timestamp=t + day * 86400 + j * 21600,
                metadata={
                    "attack": "low_and_slow",
                    "day": day,
                    "escalation_factor": round(fraction, 2),
                },
            )


# ---------------------------------------------------------------------------
# Attack 4: Amount Splitting
# ---------------------------------------------------------------------------

def amount_splitting_attack(
    target_amount: float = 100_000.0,
    n_splits: int = 10,
    user_id: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Split a large transaction into n_splits smaller ones, each below the
    high-amount threshold.

    Strategy: distribute target_amount across different merchants with
    slightly randomised amounts so no single transaction looks suspicious.

    Counter: the 24h amount velocity window should catch this.
    Evasion attempt: spread over 48h so the 24h window never sees the full sum.
    """
    uid = user_id or f"u_split_{uuid.uuid4().hex[:6]}"
    device = f"dev_{uuid.uuid4().hex[:8]}"
    t = time.time() - 48 * 3600

    # Randomise split amounts while preserving total
    weights = [random.uniform(0.5, 1.5) for _ in range(n_splits)]
    total_w = sum(weights)
    amounts = [(w / total_w) * target_amount for w in weights]

    spread_s = 48 * 3600
    for i, amt in enumerate(amounts):
        yield _txn(
            user_id=uid,
            merchant_id=f"merch_split_{i:02d}",
            amount=round(amt, 2),
            mcc=random.choice(MCCS),
            city=random.choice(CITIES),
            device_id=device,
            timestamp=t + (i / n_splits) * spread_s,
            metadata={
                "attack": "amount_splitting",
                "split_index": i,
                "target_total": target_amount,
            },
        )


# ---------------------------------------------------------------------------
# Attack 5: Geo Spoofing
# ---------------------------------------------------------------------------

def geo_spoofing_attack(
    n_txns: int = 20,
    user_id: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Simulate impossible geo-velocity without triggering the 1-hour rule.

    Rule: transactions in 2+ cities within 1 hour → flag.
    Evasion: space transactions > 1 hour apart but alternate cities so the
    overall travel pattern is physically impossible over 24h.

    Counter: ML's unique_cities_24h feature should detect the breadth.
    """
    uid = user_id or f"u_geo_{uuid.uuid4().hex[:6]}"
    device = f"dev_{uuid.uuid4().hex[:8]}"
    t = time.time() - 24 * 3600

    # Alternate between geographically distant cities every 61 minutes
    city_pairs = [("Mumbai", "Delhi"), ("Bangalore", "Kolkata"),
                  ("Chennai", "Ahmedabad"), ("Hyderabad", "Jaipur")]

    for i in range(n_txns):
        pair = city_pairs[i % len(city_pairs)]
        city = pair[i % 2]
        yield _txn(
            user_id=uid,
            merchant_id=f"merch_{random.randint(1, 20):03d}",
            amount=random.uniform(500, 5000),
            mcc=random.choice(MCCS),
            city=city,
            device_id=device,
            timestamp=t + i * 3660,   # 61-minute intervals
            metadata={
                "attack": "geo_spoofing",
                "sequence": i,
                "impossible_travel": True,
            },
        )


# ---------------------------------------------------------------------------
# Attack 6: Bot-Generated (Regular Timing)
# ---------------------------------------------------------------------------

def bot_generated_attack(
    n_txns: int = 100,
    interval_s: float = 60.0,
    amount: float = 999.0,
    user_id: Optional[str] = None,
) -> Generator[dict, None, None]:
    """
    Machine-perfect transaction timing — a strong signal of automated fraud.

    Humans have random timing; bots fire at exactly `interval_s` intervals.
    Feature: timing_regularity (std of inter-arrival times / mean).
    A legitimate user has high std; a bot has std ≈ 0.

    The amount 999 is chosen to stay just below common round-number thresholds.
    """
    uid = user_id or f"u_bot_{uuid.uuid4().hex[:6]}"
    device = f"dev_{uuid.uuid4().hex[:8]}"
    t = time.time()

    for i in range(n_txns):
        yield _txn(
            user_id=uid,
            merchant_id=f"merch_bot_{(i % 5):02d}",
            amount=amount + (i % 3) * 0.01,  # tiny variation to avoid exact duplicates
            mcc=random.choice(MCCS),
            city="Mumbai",
            device_id=device,
            timestamp=t + i * interval_s,   # perfectly regular
            metadata={
                "attack": "bot_generated",
                "interval_s": interval_s,
                "sequence": i,
            },
        )


# ---------------------------------------------------------------------------
# Attack Suite — benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class AttackResult:
    attack_name: str
    n_transactions: int
    n_blocked: int
    n_reviewed: int
    n_passed: int
    detection_rate: float       # (blocked + reviewed) / total
    block_rate: float           # blocked / total
    avg_score: float
    latencies_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "attack": self.attack_name,
            "n_transactions": self.n_transactions,
            "n_blocked": self.n_blocked,
            "n_reviewed": self.n_reviewed,
            "n_passed": self.n_passed,
            "detection_rate": round(self.detection_rate, 3),
            "block_rate": round(self.block_rate, 3),
            "avg_score": round(self.avg_score, 3),
            "p99_latency_ms": round(
                sorted(self.latencies_ms)[
                    max(0, math.ceil(0.99 * len(self.latencies_ms)) - 1)
                ] if self.latencies_ms else 0.0,
                2,
            ),
        }


@dataclass
class BenchmarkReport:
    attacks: list[AttackResult] = field(default_factory=list)
    overall_detection_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "overall_detection_rate": round(self.overall_detection_rate, 3),
            "attacks": [a.to_dict() for a in self.attacks],
        }

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"  Verity Adversarial Benchmark Report")
        print(f"  Overall detection rate: {self.overall_detection_rate:.1%}")
        print(f"{'='*60}")
        for a in self.attacks:
            print(
                f"  {a.attack_name:<22} | "
                f"detect={a.detection_rate:.1%}  "
                f"block={a.block_rate:.1%}  "
                f"avg_score={a.avg_score:.3f}"
            )
        print(f"{'='*60}\n")


class AttackSuite:
    """
    Runs all six attack patterns through a provided scoring function
    and produces a structured benchmark report.

    Parameters
    ----------
    score_fn : callable
        Takes a transaction dict, returns (risk_score, decision) tuple.
    """

    ATTACKS = [
        ("VelocityAttack",    lambda: velocity_attack(n_txns=30)),
        ("DeviceFarm",        lambda: device_farm_attack(n_users=50, n_devices=3)),
        ("LowAndSlow",        lambda: low_and_slow_attack(n_days=5)),
        ("AmountSplitting",   lambda: amount_splitting_attack(n_splits=8)),
        ("GeoSpoofing",       lambda: geo_spoofing_attack(n_txns=20)),
        ("BotGenerated",      lambda: bot_generated_attack(n_txns=40)),
    ]

    def __init__(
        self,
        score_fn: Callable[[dict], tuple[float, str]],
    ):
        self.score_fn = score_fn

    def run(self) -> BenchmarkReport:
        """Run all attacks. Returns BenchmarkReport."""
        results = []
        for name, gen_fn in self.ATTACKS:
            result = self._run_attack(name, gen_fn())
            results.append(result)

        total_txns = sum(r.n_transactions for r in results)
        total_detected = sum(r.n_blocked + r.n_reviewed for r in results)
        overall = total_detected / total_txns if total_txns > 0 else 0.0

        report = BenchmarkReport(
            attacks=results,
            overall_detection_rate=overall,
        )
        return report

    def _run_attack(
        self, name: str, transactions: Generator
    ) -> AttackResult:
        blocked = reviewed = passed = 0
        scores = []
        latencies = []

        for txn in transactions:
            t0 = time.perf_counter()
            try:
                score, decision = self.score_fn(txn)
            except Exception:
                score, decision = 0.0, "PASS"
            latency_ms = (time.perf_counter() - t0) * 1000

            scores.append(score)
            latencies.append(latency_ms)
            if decision == "BLOCK":
                blocked += 1
            elif decision == "REVIEW":
                reviewed += 1
            else:
                passed += 1

        n = blocked + reviewed + passed
        return AttackResult(
            attack_name=name,
            n_transactions=n,
            n_blocked=blocked,
            n_reviewed=reviewed,
            n_passed=passed,
            detection_rate=(blocked + reviewed) / n if n > 0 else 0.0,
            block_rate=blocked / n if n > 0 else 0.0,
            avg_score=sum(scores) / len(scores) if scores else 0.0,
            latencies_ms=latencies,
        )