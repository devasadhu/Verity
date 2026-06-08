"""
Rule engine — hard rules that fire before ML scoring.

Real fraud systems run rules first:
    - Rules are deterministic, microsecond-latency, human-auditable
    - ML handles the ambiguous middle ground rules can't cover
    - ~80% of obvious fraud is caught by rules; ML catches the rest

Rule structure:
    Each rule is a function: (txn, features, context) → RuleResult
    Rules return: PASS / BLOCK / REVIEW with a reason string

Rules are ordered by severity — BLOCK rules run first.
The first BLOCK result short-circuits; all rules run for REVIEW.

Adding a new rule: define a function, decorate with @rule("name", severity).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
import math


class Decision(str, Enum):
    PASS   = "pass"
    REVIEW = "review"
    BLOCK  = "block"


@dataclass
class RuleResult:
    rule_name: str
    decision: Decision
    reason: str
    score_contribution: float = 0.0  # additive contribution to risk score


@dataclass
class RuleEngineResult:
    final_decision: Decision
    triggered_rules: list[RuleResult]
    rule_score: float       # sum of score contributions from triggered rules
    passed_to_ml: bool      # True if ambiguous enough to need ML

    @property
    def blocked(self) -> bool:
        return self.final_decision == Decision.BLOCK

    @property
    def needs_review(self) -> bool:
        return self.final_decision == Decision.REVIEW


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------

def rule_amount_velocity(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Block if amount is more than N sigma above user's recent mean."""
    zscore = features.get("amount_zscore_1h", 0)
    mean   = features.get("user_amount_mean_1h", 0)

    if mean < 10:
        return None  # not enough history to establish baseline

    if zscore > 8:
        return RuleResult(
            rule_name="amount_velocity",
            decision=Decision.BLOCK,
            reason=f"Amount is {zscore:.1f}σ above user 1h mean (mean=₹{mean:.0f})",
            score_contribution=0.6,
        )
    if zscore > 5:
        return RuleResult(
            rule_name="amount_velocity",
            decision=Decision.REVIEW,
            reason=f"Amount is {zscore:.1f}σ above user 1h mean (mean=₹{mean:.0f})",
            score_contribution=0.3,
        )
    return None


def rule_transaction_velocity(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Block if user has too many transactions in a short window."""
    count_1m  = features.get("user_txn_count_1m", 0)
    count_5m  = features.get("user_txn_count_5m", 0)
    count_1h  = features.get("user_txn_count_1h", 0)

    if count_1m >= 5:
        return RuleResult(
            rule_name="txn_velocity_1m",
            decision=Decision.BLOCK,
            reason=f"{count_1m} transactions in last 1 minute",
            score_contribution=0.7,
        )
    if count_5m >= 12:
        return RuleResult(
            rule_name="txn_velocity_5m",
            decision=Decision.BLOCK,
            reason=f"{count_5m} transactions in last 5 minutes",
            score_contribution=0.65,
        )
    if count_1h >= 40:
        return RuleResult(
            rule_name="txn_velocity_1h",
            decision=Decision.REVIEW,
            reason=f"{count_1h} transactions in last hour",
            score_contribution=0.25,
        )
    return None


def rule_new_device_high_amount(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Review/block if new device + high amount combination."""
    is_new_device  = txn.get("is_new_device", False)
    device_age_hrs = txn.get("device_age_hours", 9999)
    amount         = float(txn.get("amount", 0))
    user_mean      = features.get("user_amount_mean_1h", 0)

    if not is_new_device and device_age_hrs > 24:
        return None  # established device

    if user_mean > 0 and amount > user_mean * 5:
        return RuleResult(
            rule_name="new_device_high_amount",
            decision=Decision.BLOCK,
            reason=(
                f"New device (age={device_age_hrs:.1f}h) + "
                f"amount ₹{amount:.0f} is {amount/user_mean:.1f}× user mean"
            ),
            score_contribution=0.55,
        )
    if is_new_device or device_age_hrs < 2:
        return RuleResult(
            rule_name="new_device",
            decision=Decision.REVIEW,
            reason=f"Transaction from new/young device (age={device_age_hrs:.1f}h)",
            score_contribution=0.2,
        )
    return None


def rule_geo_velocity(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Block if multiple cities in a short time window (impossible travel)."""
    unique_cities_1h = features.get("user_unique_cities_1h", 0)

    if unique_cities_1h >= 3:
        return RuleResult(
            rule_name="geo_velocity",
            decision=Decision.BLOCK,
            reason=f"Transactions from {unique_cities_1h} different cities in 1 hour",
            score_contribution=0.65,
        )
    if unique_cities_1h == 2:
        return RuleResult(
            rule_name="geo_velocity",
            decision=Decision.REVIEW,
            reason=f"Transactions from 2 different cities in 1 hour",
            score_contribution=0.2,
        )
    return None


def rule_high_risk_mcc(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Flag high-risk merchant category codes."""
    HIGH_RISK_MCC = {
        "6051": "Crypto/Non-Fiat currency",
        "7995": "Gambling",
        "5933": "Pawn shops",
        "6211": "Security brokers",
    }
    mcc = txn.get("merchant_category_code", "")
    if mcc in HIGH_RISK_MCC:
        return RuleResult(
            rule_name="high_risk_mcc",
            decision=Decision.REVIEW,
            reason=f"High-risk MCC {mcc}: {HIGH_RISK_MCC[mcc]}",
            score_contribution=0.15,
        )
    return None


def rule_device_sharing(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Flag if device is associated with many different users."""
    device_count_24h = features.get("device_txn_count_24h", 0)

    # High device transaction count from global sketch suggests shared/farm device
    device_global = features.get("device_global_freq", 0)
    if device_global > 500:
        return RuleResult(
            rule_name="device_farm",
            decision=Decision.BLOCK,
            reason=f"Device has {device_global} global transactions — possible device farm",
            score_contribution=0.6,
        )
    return None


def rule_off_hours_high_amount(txn: dict, features: dict, context: dict) -> Optional[RuleResult]:
    """Flag high-value transactions at unusual hours (2AM-5AM)."""
    from datetime import datetime
    ts_str = txn.get("timestamp", "")
    amount = float(txn.get("amount", 0))

    try:
        hour = datetime.fromisoformat(ts_str).hour
    except (ValueError, TypeError):
        return None

    if 2 <= hour <= 5 and amount > 10000:
        return RuleResult(
            rule_name="off_hours_high_amount",
            decision=Decision.REVIEW,
            reason=f"₹{amount:.0f} transaction at {hour:02d}:00 (off-hours)",
            score_contribution=0.2,
        )
    return None


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

ALL_RULES: list[Callable] = [
    rule_transaction_velocity,      # velocity first — clearest signal
    rule_amount_velocity,
    rule_new_device_high_amount,
    rule_geo_velocity,
    rule_device_sharing,
    rule_high_risk_mcc,
    rule_off_hours_high_amount,
]


class RuleEngine:
    """
    Runs all rules against a transaction and aggregates results.
    Short-circuits on first BLOCK (stops running remaining rules).
    Collects all REVIEW results.
    """

    def __init__(self, rules: Optional[list[Callable]] = None):
        self.rules = rules if rules is not None else ALL_RULES

    def evaluate(self, txn: dict, features: dict, context: Optional[dict] = None) -> RuleEngineResult:
        """
        Run all rules. Returns aggregated result.

        Args:
            txn:      raw transaction dict
            features: precomputed window features from SlidingWindowAggregator
            context:  optional extra context (model version, shadow mode flag, etc.)
        """
        if context is None:
            context = {}

        triggered: list[RuleResult] = []
        final_decision = Decision.PASS
        rule_score = 0.0

        for rule_fn in self.rules:
            result = rule_fn(txn, features, context)
            if result is None:
                continue

            triggered.append(result)
            rule_score += result.score_contribution

            if result.decision == Decision.BLOCK:
                final_decision = Decision.BLOCK
                break  # short-circuit on block

            if result.decision == Decision.REVIEW:
                final_decision = Decision.REVIEW
                # don't break — collect all review triggers

        rule_score = min(rule_score, 1.0)  # cap at 1.0

        # Pass to ML if: no clear block AND there's some signal
        passed_to_ml = (
            final_decision != Decision.BLOCK and
            (final_decision == Decision.REVIEW or rule_score > 0.1)
        )

        return RuleEngineResult(
            final_decision=final_decision,
            triggered_rules=triggered,
            rule_score=rule_score,
            passed_to_ml=passed_to_ml,
        )

    def add_rule(self, rule_fn: Callable):
        """Dynamically add a rule at runtime."""
        self.rules.append(rule_fn)

    def remove_rule(self, rule_name: str):
        """Remove a rule by name."""
        self.rules = [
            r for r in self.rules
            if r.__name__ != rule_name
        ]