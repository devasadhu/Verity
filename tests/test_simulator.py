import pytest
from ingestion.simulator import TransactionSimulator


def test_generates_correct_count():
    sim = TransactionSimulator(n_users=100)
    batch = sim.generate_batch(50)
    assert len(batch) == 50


def test_fraud_rate_approximate():
    sim = TransactionSimulator(n_users=500, fraud_rate=0.05, seed=0)
    batch = sim.generate_batch(2000)
    fraud_count = sum(1 for t in batch if t.is_fraud)
    fraud_rate = fraud_count / len(batch)
    # Allow wide tolerance — it's random
    assert 0.02 < fraud_rate < 0.10


def test_transaction_fields_present():
    sim = TransactionSimulator(n_users=10)
    txn = sim.generate_one()
    assert txn.transaction_id.startswith("txn_")
    assert txn.amount > 0
    assert txn.currency == "INR"
    assert txn.payer_vpa != txn.payee_vpa
    assert txn.merchant_category_code in ["5411","5812","5541","5912",
                                           "5311","4111","5999","7011",
                                           "4816","6011","5945","7922",
                                           "5047","9399","6051"]


def test_serialization():
    import json
    sim = TransactionSimulator(n_users=10)
    txn = sim.generate_one()
    serialized = sim.to_json(txn)
    data = json.loads(serialized)
    assert data["transaction_id"] == txn.transaction_id
    assert data["amount"] == txn.amount