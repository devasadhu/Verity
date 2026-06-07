"""
UPI transaction simulator.
Generates synthetic transactions with realistic distributions —
merchant categories, time-of-day patterns, device fingerprints,
geographic velocity, and behavioral anomalies.
"""

import uuid
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
import json


# Real UPI merchant category codes with realistic transaction frequency weights
MCC_DISTRIBUTION = {
    "5411": ("Grocery Stores", 0.22),
    "5812": ("Eating Places", 0.18),
    "5541": ("Service Stations", 0.10),
    "5912": ("Drug Stores", 0.08),
    "5311": ("Department Stores", 0.07),
    "4111": ("Local Transit", 0.06),
    "5999": ("Retail Misc", 0.05),
    "7011": ("Hotels/Motels", 0.04),
    "4816": ("Computer Network Services", 0.04),
    "6011": ("ATM/Cash", 0.03),
    "5945": ("Electronics", 0.03),
    "7922": ("Entertainment", 0.03),
    "5047": ("Medical Supplies", 0.02),
    "9399": ("Government Services", 0.02),
    "6051": ("Crypto/Non-Fiat", 0.01),  # high-risk MCC
}

# Indian cities with lat/lon for geo simulation
CITIES = {
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Pune":      (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
    "Jaipur":    (26.9124, 75.7873),
    "Surat":     (21.1702, 72.8311),
}

BANKS = [
    "HDFC", "SBI", "ICICI", "Axis", "Kotak",
    "PNB", "BOB", "Canara", "IDFC", "Yes"
]

DEVICE_TYPES = ["android", "ios", "web"]


@dataclass
class Transaction:
    transaction_id: str
    timestamp: str               # ISO 8601
    payer_vpa: str               # user@bank
    payee_vpa: str               # merchant@bank
    amount: float
    currency: str
    device_id: str
    device_type: str
    ip_address: str
    payer_city: str
    payer_lat: float
    payer_lon: float
    merchant_category_code: str
    merchant_category_name: str
    is_new_device: bool
    device_age_hours: float
    user_txn_count_24h: int      # injected by simulator state
    user_avg_amount_30d: float
    is_fraud: bool               # ground truth label
    fraud_type: Optional[str]    # None / "velocity" / "new_device" / "geo" / "amount"


class UserProfile:
    """Maintains per-user behavioral state for realistic simulation."""

    def __init__(self, vpa: str):
        self.vpa = vpa
        self.home_city = random.choice(list(CITIES.keys()))
        self.avg_amount = random.lognormvariate(7.0, 1.2)  # ~₹1100 mean
        self.preferred_mcc = random.choices(
            list(MCC_DISTRIBUTION.keys()),
            weights=[v[1] for v in MCC_DISTRIBUTION.values()],
            k=3
        )
        self.devices: list[str] = [f"dev_{uuid.uuid4().hex[:8]}"]
        self.txn_history: list[float] = []  # recent amounts
        self.last_city = self.home_city
        self.last_txn_time: Optional[datetime] = None
        self.txn_count_24h = 0

    def get_amount(self) -> float:
        """Lognormal around user's personal average."""
        return max(1.0, random.lognormvariate(
            __import__('math').log(max(self.avg_amount, 1)),
            0.6
        ))

    def get_device(self) -> tuple[str, bool, float]:
        """Returns (device_id, is_new_device, device_age_hours)."""
        # 3% chance of new device per transaction
        if random.random() < 0.03:
            new_dev = f"dev_{uuid.uuid4().hex[:8]}"
            self.devices.append(new_dev)
            return new_dev, True, random.uniform(0.1, 48.0)
        dev = random.choice(self.devices)
        age = random.uniform(48, 8760)  # 2 days to 1 year
        return dev, False, age


class TransactionSimulator:
    """
    Generates a stream of synthetic UPI transactions.
    Produces both normal and fraudulent transactions with
    configurable fraud rate and attack patterns.
    """

    def __init__(
        self,
        n_users: int = 1000,
        fraud_rate: float = 0.012,  # ~1.2%, close to real UPI fraud rate
        seed: int = 42
    ):
        random.seed(seed)
        self.fraud_rate = fraud_rate
        self.users = {
            f"user{i:05d}@{random.choice(BANKS).lower()}": UserProfile(
                f"user{i:05d}@{random.choice(BANKS).lower()}"
            )
            for i in range(n_users)
        }
        self.vpas = list(self.users.keys())
        self.merchant_vpas = [
            f"merchant{i:04d}@{random.choice(BANKS).lower()}"
            for i in range(200)
        ]
        self._txn_count = 0

    def _make_ip(self, city: str) -> str:
        """Generate a plausible IP for a given city (not real routing)."""
        # Indian ISP ranges (simplified)
        prefixes = {
            "Mumbai": "103.21", "Delhi": "117.196", "Bangalore": "49.37",
            "Hyderabad": "122.168", "Chennai": "182.71",
            "Kolkata": "59.90", "Pune": "103.55", "Ahmedabad": "103.26",
            "Jaipur": "103.104", "Surat": "103.195",
        }
        prefix = prefixes.get(city, "103.0")
        return f"{prefix}.{random.randint(0,255)}.{random.randint(1,254)}"

    def _inject_fraud(self, txn: Transaction, user: UserProfile) -> Transaction:
        """Modify a transaction to be fraudulent. Returns modified transaction."""
        fraud_type = random.choices(
            ["velocity", "new_device_high_amount", "geo_impossible", "amount_spike"],
            weights=[0.3, 0.3, 0.2, 0.2]
        )[0]

        if fraud_type == "velocity":
            # Many transactions in short window — just mark, rule engine catches this
            user.txn_count_24h = random.randint(25, 60)
            txn.user_txn_count_24h = user.txn_count_24h

        elif fraud_type == "new_device_high_amount":
            txn.is_new_device = True
            txn.device_age_hours = random.uniform(0.1, 6.0)
            txn.device_id = f"dev_{uuid.uuid4().hex[:8]}"
            txn.amount = user.avg_amount * random.uniform(8, 20)

        elif fraud_type == "geo_impossible":
            # Transaction from city far from last known location
            far_cities = [c for c in CITIES if c != user.last_city]
            fraud_city = random.choice(far_cities)
            txn.payer_city = fraud_city
            txn.payer_lat, txn.payer_lon = CITIES[fraud_city]
            txn.ip_address = self._make_ip(fraud_city)

        elif fraud_type == "amount_spike":
            txn.amount = user.avg_amount * random.uniform(12, 30)

        txn.is_fraud = True
        txn.fraud_type = fraud_type
        return txn

    def generate_one(self, timestamp: Optional[datetime] = None) -> Transaction:
        """Generate a single transaction."""
        if timestamp is None:
            timestamp = datetime.now()

        payer_vpa = random.choice(self.vpas)
        user = self.users[payer_vpa]

        mcc = random.choices(
            list(MCC_DISTRIBUTION.keys()),
            weights=[v[1] for v in MCC_DISTRIBUTION.values()]
        )[0]
        mcc_name = MCC_DISTRIBUTION[mcc][0]

        city = user.home_city if random.random() < 0.85 else random.choice(
            list(CITIES.keys())
        )
        lat, lon = CITIES[city]
        device_id, is_new_device, device_age = user.get_device()

        txn = Transaction(
            transaction_id=f"txn_{uuid.uuid4().hex}",
            timestamp=timestamp.isoformat(),
            payer_vpa=payer_vpa,
            payee_vpa=random.choice(self.merchant_vpas),
            amount=round(user.get_amount(), 2),
            currency="INR",
            device_id=device_id,
            device_type=random.choice(DEVICE_TYPES),
            ip_address=self._make_ip(city),
            payer_city=city,
            payer_lat=lat,
            payer_lon=lon,
            merchant_category_code=mcc,
            merchant_category_name=mcc_name,
            is_new_device=is_new_device,
            device_age_hours=round(device_age, 2),
            user_txn_count_24h=user.txn_count_24h,
            user_avg_amount_30d=round(user.avg_amount, 2),
            is_fraud=False,
            fraud_type=None,
        )

        # Inject fraud at configured rate
        if random.random() < self.fraud_rate:
            txn = self._inject_fraud(txn, user)

        # Update user state
        user.last_city = city
        user.last_txn_time = timestamp
        user.txn_history.append(txn.amount)
        if len(user.txn_history) > 100:
            user.txn_history.pop(0)
        user.txn_count_24h = min(user.txn_count_24h + 1, 100)

        self._txn_count += 1
        return txn

    def generate_batch(
        self,
        n: int,
        start_time: Optional[datetime] = None,
        tps: float = 10.0
    ) -> list[Transaction]:
        """
        Generate n transactions.
        tps controls how spread out timestamps are.
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(seconds=n / tps)

        transactions = []
        for i in range(n):
            ts = start_time + timedelta(seconds=i / tps)
            transactions.append(self.generate_one(ts))
        return transactions

    def stream(self, tps: float = 10.0):
        """
        Infinite generator that yields transactions in real time.
        Use for live simulation.
        """
        interval = 1.0 / tps
        while True:
            yield self.generate_one()
            time.sleep(interval)

    def to_json(self, txn: Transaction) -> str:
        return json.dumps(asdict(txn))