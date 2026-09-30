from pathlib import Path
import pandas as pd


# ============================================================
# Paths
# ============================================================

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

PROCESSED.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper function
# ============================================================

def add_triples(
    triples,
    source_file,
    head_column,
    relation,
    tail_column
):
    """
    Read an edge file in chunks and add its relationships
    to the KG as (head, relation, tail) triples.
    """

    print(f"\nReading {source_file.name}...")

    rows_read = 0

    for chunk in pd.read_csv(
        source_file,
        usecols=[head_column, tail_column],
        chunksize=100_000
    ):

        chunk = chunk.dropna(
            subset=[head_column, tail_column]
        )

        for head, tail in zip(
            chunk[head_column],
            chunk[tail_column]
        ):
            triples.append(
                (
                    str(head),
                    relation,
                    str(tail)
                )
            )

        rows_read += len(chunk)

    print(f"  Rows processed: {rows_read:,}")


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BUILDING BITCOIN KNOWLEDGE GRAPH")
    print("=" * 70)

    triples = []

    # --------------------------------------------------------
    # 1. Wallet -> Transaction
    # --------------------------------------------------------

    add_triples(
        triples=triples,
        source_file=RAW / "AddrTx_edgelist.csv",
        head_column="input_address",
        relation="participates_in",
        tail_column="txId"
    )

    # --------------------------------------------------------
    # 2. Transaction -> Wallet
    # --------------------------------------------------------

    add_triples(
        triples=triples,
        source_file=RAW / "TxAddr_edgelist.csv",
        head_column="txId",
        relation="has_output",
        tail_column="output_address"
    )

    # --------------------------------------------------------
    # 3. Transaction -> Transaction
    # --------------------------------------------------------

    add_triples(
        triples=triples,
        source_file=RAW / "txs_edgelist.csv",
        head_column="txId1",
        relation="money_flows_to",
        tail_column="txId2"
    )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    print("\nCreating triple table...")

    kg = pd.DataFrame(
        triples,
        columns=["head", "relation", "tail"]
    )

    print(f"Raw triples: {len(kg):,}")

    # --------------------------------------------------------
    # Remove duplicate triples
    # --------------------------------------------------------

    before = len(kg)

    kg = kg.drop_duplicates(
        subset=["head", "relation", "tail"]
    ).reset_index(drop=True)

    duplicates_removed = before - len(kg)

    print(f"Duplicate triples removed: {duplicates_removed:,}")
    print(f"Final triples: {len(kg):,}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\nTriples by relation:")

    relation_counts = (
        kg["relation"]
        .value_counts()
        .sort_index()
    )

    print(relation_counts.to_string())

    print("\nUnique entities:")

    entities = set(kg["head"]) | set(kg["tail"])

    print(f"Total unique entities: {len(entities):,}")

    # Identify entities by their occurrence in each relation
    wallet_nodes = set(
        kg.loc[
            kg["relation"] == "participates_in",
            "head"
        ]
    )

    transaction_nodes = set(
        kg.loc[
            kg["relation"] == "participates_in",
            "tail"
        ]
    )

    transaction_nodes.update(
        kg.loc[
            kg["relation"] == "money_flows_to",
            "head"
        ]
    )

    transaction_nodes.update(
        kg.loc[
            kg["relation"] == "money_flows_to",
            "tail"
        ]
    )

    print(f"Wallet entities: {len(wallet_nodes):,}")
    print(f"Transaction entities: {len(transaction_nodes):,}")

    # --------------------------------------------------------
    # Check transaction consistency
    # --------------------------------------------------------

    print("\nChecking transaction relationships...")

    money_flow_transactions = set(
        kg.loc[
            kg["relation"] == "money_flows_to",
            "head"
        ]
    )

    money_flow_transactions.update(
        kg.loc[
            kg["relation"] == "money_flows_to",
            "tail"
        ]
    )

    wallet_transaction_ids = set(
        kg.loc[
            kg["relation"] == "participates_in",
            "tail"
        ]
    )

    wallet_transaction_ids.update(
        kg.loc[
            kg["relation"] == "has_output",
            "head"
        ]
    )

    overlap = (
        money_flow_transactions
        & wallet_transaction_ids
    )

    transaction_only = (
        money_flow_transactions
        - wallet_transaction_ids
    )

    print(
        "Transactions appearing in both "
        f"wallet and money-flow relationships: {len(overlap):,}"
    )

    print(
        "Transactions appearing only in "
        f"money-flow relationships: {len(transaction_only):,}"
    )

    # --------------------------------------------------------
    # Save KG
    # --------------------------------------------------------

    output = (
        PROCESSED
        / "knowledge_graph_triples.csv"
    )

    kg.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 70)
    print("KNOWLEDGE GRAPH CREATED")
    print("=" * 70)

    print(f"\nSaved to:")
    print(output)

    print(f"\nFinal triples: {len(kg):,}")
    print(f"Unique entities: {len(entities):,}")

    print("\nFirst 10 triples:")
    print(
        kg.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()