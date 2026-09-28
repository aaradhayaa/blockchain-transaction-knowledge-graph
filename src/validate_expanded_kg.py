from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


# 1. Wallet nodes in the original wallet-to-wallet dataset

addr_addr = pd.read_csv(
    RAW / "AddrAddr_edgelist.csv",
    dtype=str
)

original_wallets = set(addr_addr["input_address"]) | set(
    addr_addr["output_address"]
)


# 2. Wallet nodes in the expanded KG

triples = pd.read_csv(
    PROCESSED / "bitcoin_triples_full.csv",
    dtype=str
)

expanded_wallets = set()

# Wallet-to-wallet relationships
wallet_edges = triples[triples["relation"] == "transacted_with"]

expanded_wallets.update(wallet_edges["head"])
expanded_wallets.update(wallet_edges["tail"])

# Wallet-to-transaction relationships
wallet_tx = triples[triples["relation"] == "participates_in"]

expanded_wallets.update(wallet_tx["head"])

# Transaction-to-wallet relationships
tx_wallet = triples[triples["relation"] == "has_output"]

expanded_wallets.update(tx_wallet["tail"])


# 3. Transaction nodes in the expanded KG

transactions = set(wallet_tx["tail"]) | set(tx_wallet["head"])


# 4. Compare counts

print("\n--- Knowledge Graph Validation ---")

print(f"Original wallet nodes: {len(original_wallets):,}")
print(f"Expanded wallet nodes: {len(expanded_wallets):,}")
print(f"Transaction nodes: {len(transactions):,}")

print(
    "Wallet node counts match:",
    len(original_wallets) == len(expanded_wallets)
)

print(
    "Original wallets missing from expanded KG:",
    len(original_wallets - expanded_wallets)
)

print(
    "Wallets in expanded KG not in original:",
    len(expanded_wallets - original_wallets)
)