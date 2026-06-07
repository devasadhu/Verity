import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from ingestion.simulator import TransactionSimulator
from ingestion.message_queue import MessageQueue

sim = TransactionSimulator(n_users=100)
mq  = MessageQueue("data/queue/transactions")

batch = sim.generate_batch(100)
for txn in batch:
    mq.publish(json.loads(sim.to_json(txn)))

fraud = sum(1 for t in batch if t.is_fraud)
print(f"Published 100 transactions")
print(f"Queue stats: {mq.stats()}")
print(f"Fraud in batch: {fraud}/100 ({fraud}%)")