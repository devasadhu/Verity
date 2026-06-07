"""
IEEE-CIS Fraud Detection dataset adapter.
Normalizes Kaggle dataset fields to Verity's transaction schema.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection
Files needed: train_transaction.csv, train_identity.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterator
import json


# IEEE-CIS columns we actually use (dataset has 394 features — we select the
# semantically meaningful ones and engineer the rest during stream processing)
TRANSACTION_COLS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "dist1", "dist2",
    "P_emaildomain", "R_emaildomain",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",  # counting features (e.g. how many addresses on card)
    "D1", "D2", "D3", "D4",      # timedelta features
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",  # match features
    "V1", "V2", "V3",            # Vesta engineered features (sample)
]

IDENTITY_COLS = [
    "TransactionID",
    "DeviceType", "DeviceInfo",
    "id_12", "id_13", "id_14", "id_15", "id_16",
    "id_17", "id_18", "id_19", "id_20",
    "id_28", "id_29", "id_30", "id_31", "id_32", "id_33",
]


@dataclass
class NormalizedTransaction:
    """
    IEEE-CIS transaction normalized to a schema compatible with Verity's
    internal format. Not identical to the UPI schema — IEEE-CIS has different
    fields — but uses the same conventions.
    """
    transaction_id: str
    timestamp_delta: int          # seconds since reference point
    amount: float
    product_code: str
    card_type: str                # visa/mastercard/etc
    card_category: str            # credit/debit
    email_domain_payer: str
    email_domain_payee: str
    device_type: Optional[str]
    device_info: Optional[str]
    addr_billing: Optional[float]
    addr_zip: Optional[float]
    dist_from_home: Optional[float]
    count_features: dict          # C1-C14 as dict
    timedelta_features: dict      # D1-D4 as dict
    match_features: dict          # M1-M9 as dict
    vesta_features: list          # V1-V3 sampled
    is_fraud: bool
    source: str = "ieee_cis"


class IEEEAdapter:
    """
    Loads and normalizes the IEEE-CIS dataset.
    Handles missing values, type coercion, and join with identity table.
    """

    def __init__(self, data_dir: str = "data/ieee_cis"):
        self.data_dir = Path(data_dir)
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> "IEEEAdapter":
        txn_path = self.data_dir / "train_transaction.csv"
        id_path = self.data_dir / "train_identity.csv"

        if not txn_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {txn_path}. "
                "Run: python scripts/download_data.py"
            )

        print(f"Loading transactions from {txn_path}...")
        # Load only the columns we need — full dataset is ~1.4GB
        available_txn = pd.read_csv(txn_path, nrows=0).columns.tolist()
        use_txn = [c for c in TRANSACTION_COLS if c in available_txn]
        txn_df = pd.read_csv(txn_path, usecols=use_txn)
        print(f"  Loaded {len(txn_df):,} transactions")

        if id_path.exists():
            print(f"Loading identity data from {id_path}...")
            available_id = pd.read_csv(id_path, nrows=0).columns.tolist()
            use_id = [c for c in IDENTITY_COLS if c in available_id]
            id_df = pd.read_csv(id_path, usecols=use_id)
            self._df = txn_df.merge(id_df, on="TransactionID", how="left")
            print(f"  Merged. Shape: {self._df.shape}")
        else:
            print("  Identity file not found — proceeding without identity features")
            self._df = txn_df

        print(f"Fraud rate: {self._df['isFraud'].mean():.4f} "
              f"({self._df['isFraud'].sum():,} fraud / {len(self._df):,} total)")
        return self

    def _normalize_row(self, row: pd.Series) -> NormalizedTransaction:
        def safe(col, default=None):
            val = row.get(col, default)
            if pd.isna(val):
                return default
            return val

        count_features = {
            f"C{i}": safe(f"C{i}", 0.0) for i in range(1, 15)
        }
        timedelta_features = {
            f"D{i}": safe(f"D{i}", -1.0) for i in range(1, 5)
        }
        match_features = {
            f"M{i}": safe(f"M{i}", "unknown") for i in range(1, 10)
        }
        vesta = [safe(f"V{i}", 0.0) for i in range(1, 4)]

        return NormalizedTransaction(
            transaction_id=f"ieee_{int(row['TransactionID'])}",
            timestamp_delta=int(safe("TransactionDT", 0)),
            amount=float(safe("TransactionAmt", 0.0)),
            product_code=str(safe("ProductCD", "unknown")),
            card_type=str(safe("card4", "unknown")).lower(),
            card_category=str(safe("card6", "unknown")).lower(),
            email_domain_payer=str(safe("P_emaildomain", "unknown")),
            email_domain_payee=str(safe("R_emaildomain", "unknown")),
            device_type=safe("DeviceType"),
            device_info=safe("DeviceInfo"),
            addr_billing=safe("addr1"),
            addr_zip=safe("addr2"),
            dist_from_home=safe("dist1"),
            count_features=count_features,
            timedelta_features=timedelta_features,
            match_features=match_features,
            vesta_features=vesta,
            is_fraud=bool(row.get("isFraud", 0)),
        )

    def iterate(self, batch_size: int = 1000) -> Iterator[list[NormalizedTransaction]]:
        """Yield transactions in batches for stream processing."""
        if self._df is None:
            raise RuntimeError("Call .load() first")

        for start in range(0, len(self._df), batch_size):
            batch = self._df.iloc[start:start + batch_size]
            yield [self._normalize_row(row) for _, row in batch.iterrows()]

    def get_class_weights(self) -> dict:
        """Returns pos/neg class weights for focal loss training."""
        if self._df is None:
            raise RuntimeError("Call .load() first")
        n_total = len(self._df)
        n_fraud = self._df["isFraud"].sum()
        n_legit = n_total - n_fraud
        return {
            "fraud_weight": n_legit / n_fraud,   # ~286x on this dataset
            "legit_weight": 1.0,
            "fraud_rate": n_fraud / n_total,
        }

    def to_json(self, txn: NormalizedTransaction) -> str:
        from dataclasses import asdict
        return json.dumps(asdict(txn))