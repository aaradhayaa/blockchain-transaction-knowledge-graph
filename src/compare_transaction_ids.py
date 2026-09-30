from pathlib import Path
import pandas as pd


RAW = Path("data/raw")


def get_unique_ids(path, column):
    print(f"Reading {path.name}...")

    ids = set()

    for chunk in pd.read_csv(path, usecols=[column], chunksize=100_000):
        ids.update(chunk[column].dropna().unique())

    print(f"  Unique IDs: {len(ids):,}")
    return ids


def main():

    print("=" * 70)
    print("TRANSACTION ID OVERLAP AUDIT")
    print("=" * 70)

    # Transactions dataset
    transaction_ids = get_unique_ids(
        RAW / "txs_features.csv",
        "txId"
    )

    # Actors dataset
    addr_tx_ids = get_unique_ids(
        RAW / "AddrTx_edgelist.csv",
        "txId"
    )

    tx_addr_ids = get_unique_ids(
        RAW / "TxAddr_edgelist.csv",
        "txId"
    )

    actors_ids = addr_tx_ids | tx_addr_ids

    print("\n" + "-" * 70)
    print("OVERLAP")
    print("-" * 70)

    overlap = transaction_ids & actors_ids

    print(f"\nTransactions dataset IDs: {len(transaction_ids):,}")
    print(f"Actors dataset transaction IDs: {len(actors_ids):,}")
    print(f"Overlap: {len(overlap):,}")

    print(
        f"Transactions dataset represented in Actors: "
        f"{len(overlap) / len(transaction_ids) * 100:.2f}%"
    )

    print(
        f"Actors transaction IDs represented in Transactions: "
        f"{len(overlap) / len(actors_ids) * 100:.2f}%"
    )

    # Separate overlap with each Actors file
    overlap_addr_tx = transaction_ids & addr_tx_ids
    overlap_tx_addr = transaction_ids & tx_addr_ids

    print("\nOverlap with AddrTx:")
    print(f"  {len(overlap_addr_tx):,}")

    print("\nOverlap with TxAddr:")
    print(f"  {len(overlap_tx_addr):,}")

    # IDs appearing in only one source
    only_transactions = transaction_ids - actors_ids
    only_actors = actors_ids - transaction_ids

    print("\nOnly in Transactions dataset:")
    print(f"  {len(only_transactions):,}")

    print("\nOnly in Actors dataset:")
    print(f"  {len(only_actors):,}")


if __name__ == "__main__":
    main()