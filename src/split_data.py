from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bitcoin_triples.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_SEED = 42


def main():
    print("Loading knowledge graph triples...")

    triples = pd.read_csv(INPUT_PATH)

    # First split: 80% training, 20% temporary
    train, temp = train_test_split(
        triples,
        test_size=0.20,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Second split: divide temporary set equally
    valid, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # Save each split
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    valid.to_csv(OUTPUT_DIR / "valid.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\nData split complete")
    print("-------------------")
    print(f"Training triples:   {len(train):,}")
    print(f"Validation triples: {len(valid):,}")
    print(f"Test triples:       {len(test):,}")
    print(f"Total triples:      {len(train) + len(valid) + len(test):,}")

    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()