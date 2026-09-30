"""
Create a leakage-controlled train / validation / test split
for transaction-level Knowledge Graph link prediction.

Final experiment:

    Transaction --money_flows_to--> Transaction

The split guarantees that every transaction entity occurring in
validation/test is already present in the training graph.

This avoids a cold-start evaluation where TransE is asked to rank
entities that it never encountered during training.

Split:
    90% training
     5% validation
     5% test

Reverse pairs are kept in the same split.
"""

from pathlib import Path
import random
import pandas as pd


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------

INPUT_FILE = Path(
    "data/processed/link_prediction/all_triples.tsv"
)

OUTPUT_DIR = Path(
    "data/processed/link_prediction"
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

SEED = 42

TRAIN_RATIO = 0.90
VALID_RATIO = 0.05
TEST_RATIO = 0.05


# -------------------------------------------------------------------
# Union-Find
# -------------------------------------------------------------------

class UnionFind:

    def __init__(self, elements):

        self.parent = {
            element: element
            for element in elements
        }

        self.rank = {
            element: 0
            for element in elements
        }

    def find(self, x):

        if self.parent[x] != x:
            self.parent[x] = self.find(
                self.parent[x]
            )

        return self.parent[x]

    def union(self, a, b):

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:

            self.parent[root_a] = root_b

        elif self.rank[root_a] > self.rank[root_b]:

            self.parent[root_b] = root_a

        else:

            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    print("=" * 70)
    print("CREATING ENTITY-COVERED LINK-PREDICTION SPLIT")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}.\n"
            f"Run prepare_triples.py first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------------
    # Load
    # ---------------------------------------------------------------

    print(f"\nLoading: {INPUT_FILE}")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        header=None,
        names=[
            "subject",
            "relation",
            "object"
        ],
        dtype=str
    )

    print(
        f"Total triples: {len(df):,}"
    )

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    if df.isna().any().any():

        raise ValueError(
            "Missing values found."
        )

    if df["relation"].nunique() != 1:

        raise ValueError(
            "Expected exactly one relation."
        )

    # ---------------------------------------------------------------
    # Create undirected pair key
    #
    # A -> B and B -> A remain together.
    # ---------------------------------------------------------------

    df["pair_key"] = df.apply(
        lambda row: tuple(
            sorted(
                [
                    row["subject"],
                    row["object"]
                ]
            )
        ),
        axis=1
    )

    # ---------------------------------------------------------------
    # Group duplicate/reverse pairs
    # ---------------------------------------------------------------

    groups = list(
        df.groupby(
            "pair_key",
            sort=False
        )
    )

    print(
        f"Unique transaction pairs: "
        f"{len(groups):,}"
    )

    # ---------------------------------------------------------------
    # Collect all entities
    # ---------------------------------------------------------------

    entities = set(
        df["subject"]
    ) | set(
        df["object"]
    )

    print(
        f"Unique entities: {len(entities):,}"
    )

    # ---------------------------------------------------------------
    # Build a spanning forest.
    #
    # Every entity will therefore occur in at least one
    # training edge.
    # ---------------------------------------------------------------

    print(
        "\nBuilding entity-covering training backbone..."
    )

    uf = UnionFind(entities)

    backbone_indices = []
    remaining_indices = []

    for index, (_, group) in enumerate(groups):

        first_row = group.iloc[0]

        subject = first_row["subject"]
        obj = first_row["object"]

        if uf.union(subject, obj):

            backbone_indices.append(index)

        else:

            remaining_indices.append(index)

    print(
        f"Backbone edges: "
        f"{len(backbone_indices):,}"
    )

    print(
        f"Remaining edges: "
        f"{len(remaining_indices):,}"
    )

    # ---------------------------------------------------------------
    # Check whether 90% training is sufficient
    # ---------------------------------------------------------------

    total_edges = len(df)

    desired_train = int(
        total_edges * TRAIN_RATIO
    )

    if len(backbone_indices) > desired_train:

        raise RuntimeError(
            "\nThe requested training ratio is too small to "
            "cover every entity.\n"
            f"Required backbone: {len(backbone_indices):,}\n"
            f"Available training budget: {desired_train:,}\n"
            "Increase TRAIN_RATIO."
        )

    # ---------------------------------------------------------------
    # Shuffle remaining groups
    # ---------------------------------------------------------------

    rng = random.Random(SEED)

    rng.shuffle(
        remaining_indices
    )

    additional_train_needed = (
        desired_train
        - len(backbone_indices)
    )

    additional_train = remaining_indices[
        :additional_train_needed
    ]

    remaining_after_train = remaining_indices[
        additional_train_needed:
    ]

    # ---------------------------------------------------------------
    # Split validation/test
    # ---------------------------------------------------------------

    valid_count = int(
        len(remaining_after_train)
        * (
            VALID_RATIO
            / (VALID_RATIO + TEST_RATIO)
        )
    )

    valid_indices = (
        remaining_after_train[:valid_count]
    )

    test_indices = (
        remaining_after_train[valid_count:]
    )

    train_indices = (
        backbone_indices
        + additional_train
    )

    # ---------------------------------------------------------------
    # Convert groups back to rows
    # ---------------------------------------------------------------

    def groups_to_dataframe(indices):

        frames = [
            groups[i][1]
            for i in indices
        ]

        if not frames:

            return pd.DataFrame(
                columns=[
                    "subject",
                    "relation",
                    "object"
                ]
            )

        result = pd.concat(
            frames,
            ignore_index=True
        )

        return result.drop(
            columns=["pair_key"]
        )

    train = groups_to_dataframe(
        train_indices
    )

    valid = groups_to_dataframe(
        valid_indices
    )

    test = groups_to_dataframe(
        test_indices
    )

    # ---------------------------------------------------------------
    # Shuffle within splits
    # ---------------------------------------------------------------

    train = train.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    valid = valid.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    test = test.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    train_file = (
        OUTPUT_DIR / "train.tsv"
    )

    valid_file = (
        OUTPUT_DIR / "valid.tsv"
    )

    test_file = (
        OUTPUT_DIR / "test.tsv"
    )

    train.to_csv(
        train_file,
        sep="\t",
        index=False,
        header=False
    )

    valid.to_csv(
        valid_file,
        sep="\t",
        index=False,
        header=False
    )

    test.to_csv(
        test_file,
        sep="\t",
        index=False,
        header=False
    )

    # ---------------------------------------------------------------
    # Entity coverage validation
    # ---------------------------------------------------------------

    train_entities = (
        set(train["subject"])
        | set(train["object"])
    )

    valid_entities = (
        set(valid["subject"])
        | set(valid["object"])
    )

    test_entities = (
        set(test["subject"])
        | set(test["object"])
    )

    unseen_valid = (
        valid_entities - train_entities
    )

    unseen_test = (
        test_entities - train_entities
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL SPLIT")
    print("=" * 70)

    total = (
        len(train)
        + len(valid)
        + len(test)
    )

    print(
        f"\nTraining:   {len(train):,} "
        f"({len(train) / total * 100:.2f}%)"
    )

    print(
        f"Validation: {len(valid):,} "
        f"({len(valid) / total * 100:.2f}%)"
    )

    print(
        f"Test:       {len(test):,} "
        f"({len(test) / total * 100:.2f}%)"
    )

    print(
        f"Total:      {total:,}"
    )

    print("\nEntity coverage:")

    print(
        f"  Training entities: "
        f"{len(train_entities):,}"
    )

    print(
        f"  Validation entities: "
        f"{len(valid_entities):,}"
    )

    print(
        f"  Test entities: "
        f"{len(test_entities):,}"
    )

    print(
        f"\nUnseen validation entities: "
        f"{len(unseen_valid):,}"
    )

    print(
        f"Unseen test entities: "
        f"{len(unseen_test):,}"
    )

    if unseen_valid or unseen_test:

        raise RuntimeError(
            "Entity coverage failed. "
            "Validation/test contains unseen entities."
        )

    print(
        "\n✓ Every validation/test entity "
        "is present in training."
    )

    # ---------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------

    summary = pd.DataFrame({
        "split": [
            "train",
            "validation",
            "test"
        ],
        "triples": [
            len(train),
            len(valid),
            len(test)
        ],
        "percentage": [
            len(train) / total * 100,
            len(valid) / total * 100,
            len(test) / total * 100
        ],
        "entities": [
            len(train_entities),
            len(valid_entities),
            len(test_entities)
        ]
    })

    summary_file = (
        OUTPUT_DIR / "split_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False
    )

    print("\nSaved:")

    print(f"  {train_file}")
    print(f"  {valid_file}")
    print(f"  {test_file}")
    print(f"  {summary_file}")

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()