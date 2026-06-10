"""
Alert Engine — fraud alert generation, deduplication, and delivery.

Decision thresholds:
    score ≥ 0.90  → BLOCK  (auto-block, alert ops team)
    score  0.70-0.90 → REVIEW (flag for human review queue)
    score < 0.70  → PASS

Alert deduplication:
    Don't fire 1000 alerts for the same compromised card.
    A (user_id, alert_type) pair is suppressed for `dedup_window_s` seconds
    after the first alert. This mirrors how PagerDuty deduplication works.

Webhook delivery:
    Alerts are delivered to registered webhook URLs via POST requests.
    Exponential backoff retry: wait 1s, 2s, 4s, 8s, 16s before giving up.
    Alerts that fail all retries go to a dead-letter queue for manual review.

No external dependencies except `urllib` (stdlib).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertType(str, Enum):
    AUTO_BLOCK        = "AUTO_BLOCK"       # score ≥ 0.90
    REVIEW_FLAGGED    = "REVIEW_FLAGGED"   # score 0.70-0.90
    DRIFT_DETECTED    = "DRIFT_DETECTED"   # PSI drift alert
    RULE_TRIGGERED    = "RULE_TRIGGERED"   # rule engine fired BLOCK
    VELOCITY_ATTACK   = "VELOCITY_ATTACK"  # burst of transactions
    COMPROMISED_CARD  = "COMPROMISED_CARD" # multiple blocks on same card


@dataclass
class Alert:
    alert_id: str
    alert_type: AlertType
    transaction_id: str
    user_id: str
    risk_score: float
    decision: str
    rule_triggers: list[str]
    top_shap: list                # [(feature, shap_value), ...]
    timestamp: float              # unix epoch
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["alert_type"] = self.alert_type.value
        return d


# ---------------------------------------------------------------------------
# Deduplication cache
# ---------------------------------------------------------------------------

class DedupCache:
    """
    Simple TTL-based deduplication cache.

    Key: (user_id, alert_type). Value: last alert timestamp.
    After `window_s` seconds the key expires and the next alert fires.
    """

    def __init__(self, window_s: float = 300.0):
        self.window_s = window_s
        self._cache: dict[tuple, float] = {}
        self._lock = threading.Lock()

    def should_suppress(self, user_id: str, alert_type: AlertType) -> bool:
        """Return True if this alert should be suppressed (dedup)."""
        key = (user_id, alert_type)
        now = time.time()
        with self._lock:
            last = self._cache.get(key)
            if last is not None and (now - last) < self.window_s:
                return True
            self._cache[key] = now
            return False

    def evict_expired(self) -> None:
        """Remove stale entries (call periodically to prevent unbounded growth)."""
        now = time.time()
        with self._lock:
            self._cache = {
                k: v for k, v in self._cache.items()
                if (now - v) < self.window_s
            }

    def size(self) -> int:
        with self._lock:
            return len(self._cache)


# ---------------------------------------------------------------------------
# Webhook delivery with exponential backoff
# ---------------------------------------------------------------------------

@dataclass
class WebhookConfig:
    url: str
    secret: str = ""             # optional HMAC secret for signing
    timeout_s: float = 3.0
    max_retries: int = 5
    initial_backoff_s: float = 1.0


def _deliver_webhook(
    alert: Alert,
    config: WebhookConfig,
) -> bool:
    """
    POST the alert to the webhook URL.
    Retry with exponential backoff on failure.
    Returns True on success, False on all retries exhausted.
    """
    payload = json.dumps(alert.to_dict()).encode()
    headers = {
        "Content-Type": "application/json",
        "X-Verity-Alert-ID": alert.alert_id,
        "X-Verity-Alert-Type": alert.alert_type.value,
    }

    backoff = config.initial_backoff_s
    for attempt in range(config.max_retries):
        try:
            req = urllib.request.Request(
                config.url,
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=config.timeout_s) as resp:
                if resp.status < 300:
                    return True
        except (urllib.error.URLError, OSError):
            pass

        # Don't sleep after the last attempt
        if attempt < config.max_retries - 1:
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)   # cap at 30s

    return False


# ---------------------------------------------------------------------------
# Alert Engine
# ---------------------------------------------------------------------------

class AlertEngine:
    """
    Generates, deduplicates, and delivers fraud alerts.

    Usage:
        engine = AlertEngine()
        engine.register_webhook(WebhookConfig(url="https://ops.example.com/alerts"))
        engine.register_handler(lambda a: print(a.alert_type, a.user_id))
        # On each scoring decision:
        engine.process(score_response, rule_triggers, shap_top_features)

    Webhook delivery runs in a background thread pool so it never blocks the
    hot path.
    """

    BLOCK_THRESHOLD  = 0.90
    REVIEW_THRESHOLD = 0.70

    def __init__(
        self,
        dedup_window_s: float = 300.0,
        n_delivery_workers: int = 2,
    ):
        self._dedup = DedupCache(window_s=dedup_window_s)
        self._webhooks: list[WebhookConfig] = []
        self._handlers: list[Callable[[Alert], None]] = []

        # Async delivery queue
        self._delivery_q: queue.Queue = queue.Queue(maxsize=10_000)
        self._dead_letter: list[Alert] = []
        self._dead_letter_lock = threading.Lock()

        # Metrics
        self._fired = 0
        self._suppressed = 0
        self._delivered = 0
        self._delivery_failed = 0

        # Start delivery workers
        for _ in range(n_delivery_workers):
            t = threading.Thread(target=self._delivery_worker, daemon=True)
            t.start()

        # Periodic dedup eviction
        t_evict = threading.Thread(target=self._evict_loop, daemon=True)
        t_evict.start()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def register_webhook(self, config: WebhookConfig) -> None:
        self._webhooks.append(config)

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a synchronous callback (called before async delivery)."""
        self._handlers.append(handler)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        transaction_id: str,
        user_id: str,
        risk_score: float,
        decision: str,
        rule_triggers: Optional[list[str]] = None,
        shap_top_features: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Optional[Alert]:
        """
        Generate an alert if warranted. Returns Alert or None (if suppressed).

        Call this for every BLOCK or REVIEW decision.
        """
        if risk_score < self.REVIEW_THRESHOLD and decision == "PASS":
            return None

        alert_type = (
            AlertType.AUTO_BLOCK if risk_score >= self.BLOCK_THRESHOLD
            else AlertType.REVIEW_FLAGGED
        )

        # Deduplication
        if self._dedup.should_suppress(user_id, alert_type):
            self._suppressed += 1
            return None

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            transaction_id=str(transaction_id),
            user_id=str(user_id),
            risk_score=float(risk_score),
            decision=decision,
            rule_triggers=list(rule_triggers or []),
            top_shap=list(shap_top_features or []),
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self._fire(alert)
        return alert

    def fire_drift_alert(
        self, drift_report: dict, metadata: Optional[dict] = None
    ) -> Alert:
        """Fire a drift alert (not subject to user-level deduplication)."""
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=AlertType.DRIFT_DETECTED,
            transaction_id="",
            user_id="system",
            risk_score=0.0,
            decision="MONITOR",
            rule_triggers=[],
            top_shap=[],
            timestamp=time.time(),
            metadata={**(metadata or {}), "drift_report": drift_report},
        )
        self._fire(alert)
        return alert

    # ------------------------------------------------------------------
    # Internal delivery
    # ------------------------------------------------------------------

    def _fire(self, alert: Alert) -> None:
        """Call sync handlers, then enqueue for async webhook delivery."""
        self._fired += 1
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception:
                pass
        try:
            self._delivery_q.put_nowait(alert)
        except queue.Full:
            # If queue is full, add to dead letter directly
            with self._dead_letter_lock:
                self._dead_letter.append(alert)

    def _delivery_worker(self) -> None:
        """Background thread: dequeue alerts and deliver to all webhooks."""
        while True:
            try:
                alert: Alert = self._delivery_q.get(timeout=1.0)
            except queue.Empty:
                continue

            all_ok = True
            for webhook in self._webhooks:
                ok = _deliver_webhook(alert, webhook)
                if ok:
                    self._delivered += 1
                else:
                    self._delivery_failed += 1
                    all_ok = False

            if not all_ok and not self._webhooks:
                pass   # no webhooks configured — normal in test mode

            if not all_ok and self._webhooks:
                with self._dead_letter_lock:
                    self._dead_letter.append(alert)

            self._delivery_q.task_done()

    def _evict_loop(self) -> None:
        while True:
            time.sleep(60)
            self._dedup.evict_expired()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "fired": self._fired,
            "suppressed": self._suppressed,
            "delivered": self._delivered,
            "delivery_failed": self._delivery_failed,
            "dead_letter_count": len(self._dead_letter),
            "queue_size": self._delivery_q.qsize(),
            "dedup_cache_size": self._dedup.size(),
        }

    def drain_dead_letter(self) -> list[Alert]:
        """Return and clear all dead-letter alerts."""
        with self._dead_letter_lock:
            alerts = list(self._dead_letter)
            self._dead_letter = []
        return alerts