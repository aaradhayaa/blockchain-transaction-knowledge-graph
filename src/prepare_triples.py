"""
Prepare the transaction-level Knowledge Graph for link prediction.

Final experiment:
    Transaction --money_flows_to--> Transaction

Input:
    data/raw/txs_edgelist.csv

Output:
    data/processed/link_prediction/all_triples.tsv

The output contains exactly three columns:
    subject    relation    object

No transaction labels or features are included because they are not
part of the link-prediction input graph.
"""

from pathlib import Path
import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

INPUT_FILE = Path("data/raw/txs_edgelist.csv")
OUTPUT_DIR = Path("data/processed/link_prediction")
OUTPUT_FILE = OUTPUT_DIR / "all_triples.tsv"


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

RELATION = "money_flows_to"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("PREPARING TRANSACTION LINK-PREDICTION TRIPLES")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    print(f"\nLoading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        dtype=str
    )

    expected_columns = {"txId1", "txId2"}

    if not expected_columns.issubset(df.columns):
        raise ValueError(
            f"Expected columns {expected_columns}, "
            f"but found {list(df.columns)}"
        )

    print(f"Input edges: {len(df):,}")

    # ---------------------------------------------------------------
    # Clean
    # ---------------------------------------------------------------

    df = df[["txId1", "txId2"]].copy()

    df["txId1"] = df["txId1"].str.strip()
    df["txId2"] = df["txId2"].str.strip()

    # Remove missing IDs
    before = len(df)

    df = df.dropna(subset=["txId1", "txId2"])

    print(
        f"Removed missing-ID rows: "
        f"{before - len(df):,}"
    )

    # Remove empty IDs
    before = len(df)

    df = df[
        (df["txId1"] != "") &
        (df["txId2"] != "")
    ]

    print(
        f"Removed empty-ID rows: "
        f"{before - len(df):,}"
    )

    # Remove self-loops
    before = len(df)

    df = df[df["txId1"] != df["txId2"]]

    print(
        f"Removed self-loops: "
        f"{before - len(df):,}"
    )

    # Remove exact duplicate edges
    before = len(df)

    df = df.drop_duplicates(
        subset=["txId1", "txId2"]
    )

    print(
        f"Removed duplicate edges: "
        f"{before - len(df):,}"
    )

    # ---------------------------------------------------------------
    # Convert to KG triples
    # ---------------------------------------------------------------

    triples = pd.DataFrame({
        "subject": df["txId1"],
        "relation": RELATION,
        "object": df["txId2"],
    })

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    entities = set(triples["subject"]) | set(triples["object"])

    print("\n" + "=" * 70)
    print("FINAL TRANSACTION KG")
    print("=" * 70)

    print(f"Entities: {len(entities):,}")
    print(f"Relations: 1")
    print(f"Triples: {len(triples):,}")
    print(f"Relation: {RELATION}")

    print("\nThis KG contains:")
    print("  Transaction -> money_flows_to -> Transaction")

    print("\nExcluded from the embedding experiment:")
    print("  Wallet relations")
    print("  has_output")
    print("  AddrAddr")
    print("  Transaction features")
    print("  Transaction classes")

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    # PyKEEN-compatible TSV: no header.
    triples.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
        header=False
    )

    print(f"\nSaved:")
    print(f"  {OUTPUT_FILE}")

    print("\n" + "=" * 70)
    print("PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()