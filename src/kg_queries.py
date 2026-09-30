from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

EDGE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "all_triples.tsv"
)

RESULTS_DIR = BASE_DIR / "results" / "kg_queries"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD KG
# ============================================================

def load_kg():

    df = pd.read_csv(
        EDGE_FILE,
        sep="\t",
        header=None,
        names=["source", "relation", "target"],
        dtype=str,
    )

    return df


# ============================================================
# QUERY FUNCTIONS
# ============================================================

def outgoing_transactions(df, transaction_id):

    return df[
        (df["source"] == transaction_id)
        & (df["relation"] == "money_flows_to")
    ]["target"].tolist()


def incoming_transactions(df, transaction_id):

    return df[
        (df["target"] == transaction_id)
        & (df["relation"] == "money_flows_to")
    ]["source"].tolist()


def downstream_two_hop(df, transaction_id):

    first_hop = set(
        outgoing_transactions(
            df,
            transaction_id
        )
    )

    second_hop = set()

    for transaction in first_hop:

        second_hop.update(
            outgoing_transactions(
                df,
                transaction
            )
        )

    # Remove the starting transaction
    second_hop.discard(transaction_id)

    # Remove direct neighbours
    second_hop -= first_hop

    return sorted(second_hop)


def transaction_degree(df, transaction_id):

    outgoing = len(
        outgoing_transactions(
            df,
            transaction_id
        )
    )

    incoming = len(
        incoming_transactions(
            df,
            transaction_id
        )
    )

    return incoming, outgoing


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KNOWLEDGE GRAPH QUERY / SERVICE DEMONSTRATION")
    print("=" * 70)

    print("\nLoading knowledge graph...")

    df = load_kg()

    print(
        f"Loaded {len(df):,} triples."
    )

    # --------------------------------------------------------
    # Select a transaction that actually has outgoing edges
    # --------------------------------------------------------

    flow_edges = df[
        df["relation"] == "money_flows_to"
    ]

    if flow_edges.empty:
        raise RuntimeError(
            "No money_flows_to relationships found."
        )

    # Choose a transaction with at least one outgoing
    # and one downstream relationship if possible.

    outgoing_counts = (
        flow_edges
        .groupby("source")
        .size()
        .sort_values(ascending=False)
    )

    transaction_id = outgoing_counts.index[0]

    # --------------------------------------------------------
    # Query 1: outgoing relationships
    # --------------------------------------------------------

    outgoing = outgoing_transactions(
        df,
        transaction_id
    )

    print("\n" + "-" * 70)
    print("QUERY 1: DOWNSTREAM TRANSACTIONS")
    print("-" * 70)

    print(
        f"\nTransaction: {transaction_id}"
    )

    print(
        f"Direct downstream transactions: "
        f"{len(outgoing)}"
    )

    for target in outgoing[:10]:
        print(f"  -> {target}")

    if len(outgoing) > 10:
        print(
            f"  ... and {len(outgoing) - 10:,} more"
        )

    # --------------------------------------------------------
    # Query 2: incoming relationships
    # --------------------------------------------------------

    incoming = incoming_transactions(
        df,
        transaction_id
    )

    print("\n" + "-" * 70)
    print("QUERY 2: UPSTREAM TRANSACTIONS")
    print("-" * 70)

    print(
        f"\nTransaction: {transaction_id}"
    )

    print(
        f"Direct upstream transactions: "
        f"{len(incoming)}"
    )

    for source in incoming[:10]:
        print(f"  <- {source}")

    if len(incoming) > 10:
        print(
            f"  ... and {len(incoming) - 10:,} more"
        )

    # --------------------------------------------------------
    # Query 3: two-hop downstream
    # --------------------------------------------------------

    two_hop = downstream_two_hop(
        df,
        transaction_id
    )

    print("\n" + "-" * 70)
    print("QUERY 3: TWO-HOP DOWNSTREAM TRANSACTIONS")
    print("-" * 70)

    print(
        f"\nTransaction: {transaction_id}"
    )

    print(
        f"Two-hop downstream transactions: "
        f"{len(two_hop)}"
    )

    for target in two_hop[:10]:
        print(f"  => {target}")

    if len(two_hop) > 10:
        print(
            f"  ... and {len(two_hop) - 10:,} more"
        )

    # --------------------------------------------------------
    # Query 4: transaction degree
    # --------------------------------------------------------

    incoming_count, outgoing_count = (
        transaction_degree(
            df,
            transaction_id
        )
    )

    print("\n" + "-" * 70)
    print("QUERY 4: TRANSACTION CONNECTIVITY")
    print("-" * 70)

    print(
        f"\nTransaction: {transaction_id}"
    )

    print(
        f"Incoming relationships: "
        f"{incoming_count}"
    )

    print(
        f"Outgoing relationships: "
        f"{outgoing_count}"
    )

    # --------------------------------------------------------
    # Save query results
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "query": [
                "direct_downstream",
                "direct_upstream",
                "two_hop_downstream",
                "incoming_degree",
                "outgoing_degree",
            ],
            "transaction": [
                transaction_id,
                transaction_id,
                transaction_id,
                transaction_id,
                transaction_id,
            ],
            "result_count": [
                len(outgoing),
                len(incoming),
                len(two_hop),
                incoming_count,
                outgoing_count,
            ],
        }
    )

    output_file = (
        RESULTS_DIR / "query_results.csv"
    )

    summary.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("QUERY DEMONSTRATION COMPLETE")
    print("=" * 70)

    print(
        f"\nResults saved to:\n{output_file}"
    )


if __name__ == "__main__":
    main()