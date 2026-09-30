from pathlib import Path
import pandas as pd


RAW = Path("data/raw")


def get_ids(path, column):
    ids = set()

    for chunk in pd.read_csv(
        path,
        usecols=[column],
        chunksize=100_000
    ):
        ids.update(chunk[column].dropna().unique())

    return ids


def main():

    print("=" * 70)
    print("TRANSACTION COVERAGE ANALYSIS")
    print("=" * 70)

    # All transaction IDs
    transaction_ids = get_ids(
        RAW / "txs_features.csv",
        "txId"
    )

    # Transaction IDs connected to wallets
    addr_tx_ids = get_ids(
        RAW / "AddrTx_edgelist.csv",
        "txId"
    )

    tx_addr_ids = get_ids(
        RAW / "TxAddr_edgelist.csv",
        "txId"
    )

    actors_ids = addr_tx_ids | tx_addr_ids

    only_transactions = transaction_ids - actors_ids

    print(f"\nTransactions dataset: {len(transaction_ids):,}")
    print(f"Actors transactions: {len(actors_ids):,}")
    print(f"Transactions without Actors links: {len(only_transactions):,}")

    print("\nFirst 20 uncovered transaction IDs:")
    print(sorted(only_transactions)[:20])

    # Check whether uncovered IDs occur in transaction graph
    edges = pd.read_csv(RAW / "txs_edgelist.csv")

    source = edges.columns[0]
    target = edges.columns[1]

    source_ids = set(edges[source])
    target_ids = set(edges[target])

    uncovered_in_edges = only_transactions & (source_ids | target_ids)

    print(
        "\nUncovered transactions appearing in tx→tx graph: "
        f"{len(uncovered_in_edges):,}"
    )

    # Check classes
    classes = pd.read_csv(RAW / "txs_classes.csv")

    uncovered_classes = classes[
        classes["txId"].isin(only_transactions)
    ]

    print("\nClasses of uncovered transactions:")
    print(
        uncovered_classes["class"]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()