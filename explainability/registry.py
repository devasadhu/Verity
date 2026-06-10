"""
Model Registry + Audit Log

Model Registry: versioned JSON manifest tracking every trained model —
  metadata, hyperparameters, training metrics, artifact paths, active status.
  Models are registered with a content-hash-derived version string.

Audit Log: immutable append-only log of every scoring decision.
  Each entry records: transaction_id, timestamp, model_version,
  feature_vector hash, rule_score, ml_score, final_decision,
  SHAP top features, latency_ms.
  Append-only — entries cannot be modified or deleted. Required for
  RBI regulatory compliance framing.

Both are backed by flat files (JSON manifest + JSONL audit log) so the
system is self-contained with zero external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelRecord:
    """Metadata for one trained model version."""
    version: str                  # e.g. "mlp-20240601-a3f2"
    model_type: str               # "mlp" | "gbt" | "ensemble"
    artifact_path: str            # path to .npz or pickle
    registered_at: str            # ISO 8601
    training_dataset: str         # e.g. "ieee-cis-v1" | "upi-synthetic-v3"
    hyperparams: dict = field(default_factory=dict)
    train_metrics: dict = field(default_factory=dict)   # auc, f1, etc.
    is_active: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelRecord":
        return cls(**d)


@dataclass
class AuditEntry:
    """One scoring decision. Immutable once written."""
    entry_id: str
    transaction_id: str
    timestamp: str               # ISO 8601 UTC
    model_version: str
    feature_hash: str            # SHA-256 of serialised feature vector
    rule_score: float
    ml_score: float
    ensemble_score: float
    decision: str                # "BLOCK" | "REVIEW" | "PASS"
    top_shap_features: list      # [(feature_name, shap_value), ...]
    latency_ms: float
    shadow_mode: bool = False    # True → decision logged but not enforced
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AuditEntry":
        return cls(**d)


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Versioned model manifest stored as a JSON file.

    Thread-safe with a re-entrant lock.
    """

    def __init__(self, manifest_path: str = "models/registry.json"):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._records: dict[str, ModelRecord] = {}
        self._load()

    # --- Persistence --------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                raw = json.load(f)
            self._records = {
                k: ModelRecord.from_dict(v) for k, v in raw.items()
            }

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(
                {k: v.to_dict() for k, v in self._records.items()},
                f,
                indent=2,
            )
        tmp.replace(self.path)   # atomic rename

    # --- Public API ---------------------------------------------------

    def register(
        self,
        model_type: str,
        artifact_path: str,
        training_dataset: str,
        hyperparams: Optional[dict] = None,
        train_metrics: Optional[dict] = None,
        notes: str = "",
    ) -> ModelRecord:
        """
        Register a new model. Returns the ModelRecord with its assigned version.

        Version string: "{model_type}-{date}-{hash4}"
        where hash4 is the first 4 hex chars of the artifact file's SHA-256.
        """
        with self._lock:
            file_hash = _file_hash(artifact_path)[:4]
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            version = f"{model_type}-{date_str}-{file_hash}"

            # Handle rare collisions
            base = version
            idx = 1
            while version in self._records:
                version = f"{base}-{idx}"
                idx += 1

            record = ModelRecord(
                version=version,
                model_type=model_type,
                artifact_path=str(artifact_path),
                registered_at=_utcnow(),
                training_dataset=training_dataset,
                hyperparams=hyperparams or {},
                train_metrics=train_metrics or {},
                is_active=False,
                notes=notes,
            )
            self._records[version] = record
            self._save()
            return record

    def activate(self, version: str) -> None:
        """
        Set a model version as the active serving model.
        Deactivates the previously active version of the same model_type.
        """
        with self._lock:
            if version not in self._records:
                raise KeyError(f"Version '{version}' not found in registry.")
            target_type = self._records[version].model_type
            for rec in self._records.values():
                if rec.model_type == target_type and rec.is_active:
                    rec.is_active = False
            self._records[version].is_active = True
            self._save()

    def get_active(self, model_type: str) -> Optional[ModelRecord]:
        """Return the currently active model for a given type, or None."""
        with self._lock:
            for rec in self._records.values():
                if rec.model_type == model_type and rec.is_active:
                    return rec
            return None

    def get(self, version: str) -> Optional[ModelRecord]:
        with self._lock:
            return self._records.get(version)

    def list_all(self) -> list[ModelRecord]:
        with self._lock:
            return list(self._records.values())

    def list_by_type(self, model_type: str) -> list[ModelRecord]:
        with self._lock:
            return [r for r in self._records.values() if r.model_type == model_type]

    def update_metrics(self, version: str, metrics: dict) -> None:
        """Add/update training metrics for a registered model."""
        with self._lock:
            if version not in self._records:
                raise KeyError(version)
            self._records[version].train_metrics.update(metrics)
            self._save()


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Immutable append-only log of every scoring decision.

    Backed by a JSONL file (one JSON object per line).
    Thread-safe. Entries are never modified or deleted.

    In-memory index on transaction_id for fast lookups (rebuilt on load).
    """

    def __init__(self, log_path: str = "models/audit.jsonl"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index: dict[str, list[AuditEntry]] = {}  # txn_id → entries
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = AuditEntry.from_dict(json.loads(line))
                    self._index.setdefault(entry.transaction_id, []).append(entry)
                except (json.JSONDecodeError, TypeError):
                    continue   # corrupted line — skip

    def log(
        self,
        transaction_id: str,
        model_version: str,
        feature_vector,           # list or np.ndarray
        rule_score: float,
        ml_score: float,
        ensemble_score: float,
        decision: str,
        top_shap_features: list,
        latency_ms: float,
        shadow_mode: bool = False,
        metadata: Optional[dict] = None,
    ) -> AuditEntry:
        """Append one decision to the audit log. Returns the AuditEntry."""
        entry = AuditEntry(
            entry_id=str(uuid.uuid4()),
            transaction_id=str(transaction_id),
            timestamp=_utcnow(),
            model_version=model_version,
            feature_hash=_feature_hash(feature_vector),
            rule_score=float(rule_score),
            ml_score=float(ml_score),
            ensemble_score=float(ensemble_score),
            decision=decision,
            top_shap_features=list(top_shap_features),
            latency_ms=float(latency_ms),
            shadow_mode=shadow_mode,
            metadata=metadata or {},
        )
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())   # durability
            self._index.setdefault(entry.transaction_id, []).append(entry)
        return entry

    def get(self, transaction_id: str) -> list[AuditEntry]:
        """Return all audit entries for a given transaction (usually 1)."""
        with self._lock:
            return list(self._index.get(transaction_id, []))

    def get_latest(self, transaction_id: str) -> Optional[AuditEntry]:
        entries = self.get(transaction_id)
        return entries[-1] if entries else None

    def stats(self) -> dict:
        """Return aggregate decision counts across the log."""
        with self._lock:
            all_entries = [e for entries in self._index.values() for e in entries]
        counts: dict[str, int] = {}
        shadow = 0
        for e in all_entries:
            counts[e.decision] = counts.get(e.decision, 0) + 1
            if e.shadow_mode:
                shadow += 1
        return {
            "total": len(all_entries),
            "by_decision": counts,
            "shadow_mode_entries": shadow,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: str) -> str:
    """SHA-256 of a file, hex-encoded. Returns zeros if file not found."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return "0" * 64


def _feature_hash(feature_vector) -> str:
    """SHA-256 of a serialised feature vector."""
    serialised = json.dumps(
        [round(float(v), 8) for v in feature_vector],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialised).hexdigest()