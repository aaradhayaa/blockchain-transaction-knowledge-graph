from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


# --------------------------------------------------
# 2. Load edge files
# --------------------------------------------------

print("Loading AddrAddr...")

addr_addr = pd.read_csv(
    RAW / "AddrAddr_edgelist.csv",
    dtype=str,
)

print("Loading AddrTx...")

addr_tx = pd.read_csv(
    RAW / "AddrTx_edgelist.csv",
    dtype=str,
)

print("Loading TxAddr...")

tx_addr = pd.read_csv(
    RAW / "TxAddr_edgelist.csv",
    dtype=str,
)


# --------------------------------------------------
# 3. Clean duplicates
# --------------------------------------------------

addr_addr = addr_addr.drop_duplicates()
addr_tx = addr_tx.drop_duplicates()
tx_addr = tx_addr.drop_duplicates()


# --------------------------------------------------
# 4. Create a set of direct wallet-to-wallet edges
# --------------------------------------------------

direct_edges = set(
    zip(
        addr_addr["input_address"],
        addr_addr["output_address"],
    )
)

print(f"\nUnique AddrAddr edges: {len(direct_edges):,}")


# --------------------------------------------------
# 5. Connect AddrTx and TxAddr through transaction IDs
# --------------------------------------------------

print("\nJoining AddrTx and TxAddr on transaction ID...")

transaction_paths = addr_tx.merge(
    tx_addr,
    on="txId",
    how="inner",
)

# Each row now represents a wallet pair
# connected through a shared transaction ID.

derived_edges = set(
    zip(
        transaction_paths["input_address"],
        transaction_paths["output_address"],
    )
)

print(
    "Unique wallet pairs derived through transactions: "
    f"{len(derived_edges):,}"
)


# --------------------------------------------------
# 6. Compare direct and transaction-derived edges
# --------------------------------------------------

overlap = direct_edges & derived_edges

print("\n--- Comparison Results ---")

print(f"Direct AddrAddr edges: {len(direct_edges):,}")

print(f"Transaction-derived pairs: {len(derived_edges):,}")

print(f"Overlapping wallet pairs: {len(overlap):,}")

if direct_edges:
    print(
        "Percentage of AddrAddr edges represented "
        "by transaction paths: "
        f"{len(overlap) / len(direct_edges) * 100:.2f}%"
    )

if derived_edges:
    print(
        "Percentage of transaction-derived pairs "
        "also found in AddrAddr: "
        f"{len(overlap) / len(derived_edges) * 100:.2f}%"
    )


# --------------------------------------------------
# 7. Save overlap results
# --------------------------------------------------

output_path = ROOT / "data" / "processed" / "edge_source_overlap.txt"

with open(output_path, "w") as f:
    f.write(f"Unique AddrAddr edges: {len(direct_edges):,}\n")
    f.write(f"Transaction-derived pairs: {len(derived_edges):,}\n")
    f.write(f"Overlapping wallet pairs: {len(overlap):,}\n")

print(f"\nSummary saved to: {output_path}")
