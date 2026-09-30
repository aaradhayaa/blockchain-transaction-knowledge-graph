from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "train.tsv"
)

VALID_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "valid.tsv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "test.tsv"
)

RESULTS_DIR = BASE_DIR / "results" / "logical_reasoning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD GRAPH
# ============================================================

def load_edges(path):
    """Load a TSV knowledge-graph edge file."""

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["source", "relation", "target"],
        dtype=str,
    )

    return set(
        zip(
            df["source"],
            df["target"],
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOGICAL REASONING: TWO-HOP TRANSACTION PATHS")
    print("=" * 70)

    print("\nResearch rule:")
    print("  A -> B AND B -> C  =>  A -> C")
    print("\nRelation:")
    print("  money_flows_to")

    # --------------------------------------------------------
    # Load training and test edges
    # --------------------------------------------------------

    print("\nLoading training graph...")

    train_edges = load_edges(TRAIN_FILE)

    print(f"Training edges: {len(train_edges):,}")

    print("\nLoading validation graph...")

    valid_edges = load_edges(VALID_FILE)

    print(f"Validation edges: {len(valid_edges):,}")

    print("\nLoading test graph...")

    test_edges = load_edges(TEST_FILE)

    print(f"Test edges: {len(test_edges):,}")

    # --------------------------------------------------------
    # Build adjacency list
    # --------------------------------------------------------

    print("\nBuilding transaction adjacency structure...")

    outgoing = {}

    for source, target in train_edges:

        if source not in outgoing:
            outgoing[source] = set()

        outgoing[source].add(target)

    # --------------------------------------------------------
    # Generate logical two-hop conclusions
    # --------------------------------------------------------

    print("\nGenerating two-hop logical paths...")

    inferred_pairs = set()

    for source, middle in train_edges:

        next_nodes = outgoing.get(middle)

        if not next_nodes:
            continue

        for target in next_nodes:

            if source == target:
                continue

            inferred_pairs.add(
                (source, target)
            )

    print(
        f"Unique two-hop inferred relationships: "
        f"{len(inferred_pairs):,}"
    )

    # --------------------------------------------------------
    # Remove relationships already directly observed
    # --------------------------------------------------------

    novel_inferences = (
        inferred_pairs
        - train_edges
    )

    print(
        f"Novel two-hop relationships: "
        f"{len(novel_inferences):,}"
    )

    # --------------------------------------------------------
    # Compare against held-out test set
    # --------------------------------------------------------

    supported_test_edges = (
        test_edges
        & novel_inferences
    )

    support_count = len(supported_test_edges)

    coverage = (
        support_count / len(test_edges)
        if len(test_edges) > 0
        else 0
    )

    # --------------------------------------------------------
    # Test precision
    #
    # Of the logical conclusions, how many are actually
    # present in the held-out test set?
    #
    # This is a very strict measure because the test set
    # contains only a small subset of all true relationships.
    # --------------------------------------------------------

    precision_against_test = (
        support_count / len(novel_inferences)
        if len(novel_inferences) > 0
        else 0
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOGICAL REASONING RESULTS")
    print("=" * 70)

    print(
        f"\nTest relationships: "
        f"{len(test_edges):,}"
    )

    print(
        f"Test relationships supported by a "
        f"two-hop path: "
        f"{support_count:,}"
    )

    print(
        f"Test coverage: "
        f"{coverage * 100:.4f}%"
    )

    print(
        f"\nTotal novel logical conclusions: "
        f"{len(novel_inferences):,}"
    )

    print(
        f"Novel conclusions also appearing in test: "
        f"{support_count:,}"
    )

    print(
        f"Precision against test set: "
        f"{precision_against_test * 100:.4f}%"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "metric": [
                "training_edges",
                "validation_edges",
                "test_edges",
                "two_hop_inferred_relationships",
                "novel_two_hop_relationships",
                "test_edges_supported_by_two_hop_logic",
                "test_coverage_percent",
                "precision_against_test_percent",
            ],
            "value": [
                len(train_edges),
                len(valid_edges),
                len(test_edges),
                len(inferred_pairs),
                len(novel_inferences),
                support_count,
                coverage * 100,
                precision_against_test * 100,
            ],
        }
    )

    output_file = (
        RESULTS_DIR
        / "logical_reasoning_results.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Save examples
    # --------------------------------------------------------

    examples = list(
        supported_test_edges
    )[:100]

    example_df = pd.DataFrame(
        examples,
        columns=[
            "source_transaction",
            "target_transaction",
        ],
    )

    example_file = (
        RESULTS_DIR
        / "logical_test_examples.csv"
    )

    example_df.to_csv(
        example_file,
        index=False,
    )

    print("\nSaved:")
    print(output_file)
    print(example_file)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()