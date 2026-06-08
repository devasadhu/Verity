import pytest
import time
from stream.sketch import CountMinSketch
from stream.windows import SlidingWindowAggregator, EntityWindow
from stream.rules import RuleEngine, Decision


class TestCountMinSketch:
    def test_update_and_query(self):
        cms = CountMinSketch(epsilon=0.01, delta=0.001)
        for _ in range(100):
            cms.update("user:001")
        for _ in range(50):
            cms.update("user:002")
        assert cms.query("user:001") >= 100
        assert cms.query("user:002") >= 50
        assert cms.query("user:001") >= cms.query("user:002")

    def test_never_underestimates(self):
        cms = CountMinSketch(epsilon=0.01, delta=0.001)
        true_counts = {}
        import random
        random.seed(42)
        keys = [f"key_{i}" for i in range(50)]
        for _ in range(1000):
            k = random.choice(keys)
            cms.update(k)
            true_counts[k] = true_counts.get(k, 0) + 1
        for k, true_count in true_counts.items():
            assert cms.query(k) >= true_count  # never underestimates

    def test_total(self):
        cms = CountMinSketch()
        cms.update("a", 10)
        cms.update("b", 5)
        assert cms.total() == 15

    def test_reset(self):
        cms = CountMinSketch()
        cms.update("key", 100)
        cms.reset()
        assert cms.query("key") == 0
        assert cms.total() == 0


class TestEntityWindow:
    def test_basic_stats(self):
        win = EntityWindow()
        now = time.time()
        win.add(100.0, ts=now - 30)   # 30s ago — within 1m window
        win.add(200.0, ts=now - 30)
        stats = win.stats("1m", now)
        assert stats.count == 2
        assert stats.amount_mean == 150.0
        assert stats.amount_max == 200.0

    def test_window_expiry(self):
        win = EntityWindow()
        now = time.time()
        win.add(500.0, ts=now - 400)  # 6m40s ago — outside 5m, inside 1h
        stats_5m = win.stats("5m", now)
        stats_1h = win.stats("1h", now)
        assert stats_5m.count == 0
        assert stats_1h.count == 1

    def test_empty_window(self):
        win = EntityWindow()
        stats = win.stats("1m")
        assert stats.count == 0
        assert stats.amount_mean == 0


class TestSlidingWindowAggregator:
    def _make_txn(self, user, amount, device="dev_001", city="Mumbai"):
        return {
            "payer_vpa": user,
            "payee_vpa": "merchant@sbi",
            "device_id": device,
            "payer_city": city,
            "amount": amount,
            "timestamp": "2024-01-15T10:00:00",
        }

    def test_record_and_get_features(self):
        agg = SlidingWindowAggregator()
        now = time.time()
        txn = self._make_txn("user001@hdfc", 1500.0)
        agg.record(txn, ts=now)
        features = agg.get_features(txn, now=now)
        assert features["user_txn_count_1m"] == 1
        assert features["user_amount_mean_1h"] == 1500.0

    def test_velocity_accumulates(self):
        agg = SlidingWindowAggregator()
        now = time.time()
        for i in range(5):
            agg.record(self._make_txn("user001@hdfc", 100.0), ts=now - i)
        features = agg.get_features(self._make_txn("user001@hdfc", 100.0), now=now)
        assert features["user_txn_count_1m"] == 5

    def test_zscore_computation(self):
        agg = SlidingWindowAggregator()
        now = time.time()
        # Establish baseline: 10 transactions of ₹100
        for i in range(10):
            agg.record(self._make_txn("user001@hdfc", 100.0), ts=now - i * 60)
        # Now record a large transaction
        big_txn = self._make_txn("user001@hdfc", 10000.0)
        agg.record(big_txn, ts=now)
        features = agg.get_features(big_txn, now=now)
        # Z-score should be strongly positive
        assert features["amount_zscore_1h"] > 3


class TestRuleEngine:
    def _base_txn(self):
        return {
            "amount": 1000.0,
            "payer_vpa": "user001@hdfc",
            "device_id": "dev_001",
            "is_new_device": False,
            "device_age_hours": 720,
            "merchant_category_code": "5411",
            "timestamp": "2024-01-15T14:00:00",
        }

    def _base_features(self):
        return {
            "user_txn_count_1m": 1,
            "user_txn_count_5m": 2,
            "user_txn_count_1h": 5,
            "user_txn_count_24h": 20,
            "user_amount_mean_1h": 1000.0,
            "user_amount_std_1h": 200.0,
            "user_amount_max_24h": 2000.0,
            "amount_zscore_1h": 0.0,
            "user_unique_cities_1h": 1,
            "user_unique_devices_24h": 1,
            "device_txn_count_1h": 5,
            "device_txn_count_24h": 20,
            "device_global_freq": 50,
            "merchant_txn_count_1h": 10,
            "merchant_amount_mean_1h": 1000.0,
            "user_global_freq": 100,
            "merchant_global_freq": 200,
        }

    def test_normal_transaction_passes(self):
        engine = RuleEngine()
        result = engine.evaluate(self._base_txn(), self._base_features())
        assert result.final_decision == Decision.PASS

    def test_velocity_block(self):
        engine = RuleEngine()
        features = self._base_features()
        features["user_txn_count_1m"] = 10  # way above threshold
        result = engine.evaluate(self._base_txn(), features)
        assert result.final_decision == Decision.BLOCK
        assert result.blocked

    def test_amount_spike_review(self):
        engine = RuleEngine()
        txn = self._base_txn()
        txn["amount"] = 20000.0
        features = self._base_features()
        features["amount_zscore_1h"] = 6.0
        features["user_amount_mean_1h"] = 1000.0
        result = engine.evaluate(txn, features)
        assert result.final_decision in (Decision.REVIEW, Decision.BLOCK)

    def test_new_device_flag(self):
        engine = RuleEngine()
        txn = self._base_txn()
        txn["is_new_device"] = True
        txn["device_age_hours"] = 1.0
        result = engine.evaluate(txn, self._base_features())
        assert result.final_decision in (Decision.REVIEW, Decision.BLOCK)

    def test_geo_velocity_block(self):
        engine = RuleEngine()
        features = self._base_features()
        features["user_unique_cities_1h"] = 4
        result = engine.evaluate(self._base_txn(), features)
        assert result.final_decision == Decision.BLOCK

    def test_high_risk_mcc_review(self):
        engine = RuleEngine()
        txn = self._base_txn()
        txn["merchant_category_code"] = "6051"  # crypto
        result = engine.evaluate(txn, self._base_features())
        assert result.final_decision == Decision.REVIEW

    def test_rule_score_capped(self):
        engine = RuleEngine()
        features = self._base_features()
        features["user_txn_count_1m"] = 20
        features["amount_zscore_1h"] = 10.0
        features["user_amount_mean_1h"] = 500.0
        features["user_unique_cities_1h"] = 5
        result = engine.evaluate(self._base_txn(), features)
        assert result.rule_score <= 1.0