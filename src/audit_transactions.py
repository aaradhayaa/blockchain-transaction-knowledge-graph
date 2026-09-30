from pathlib import Path
import pandas as pd


RAW = Path("data/raw")


def inspect_file(path, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Read only the header
    df_head = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    print(list(df_head.columns))

    print("\nFirst 5 rows:")
    print(df_head.to_string(index=False))

    # Count rows without loading the whole file
    rows = 0
    for chunk in pd.read_csv(path, chunksize=100_000):
        rows += len(chunk)

    print(f"\nTotal rows: {rows}")


def inspect_edge_file(path):
    print("\n" + "=" * 70)
    print("TRANSACTION EDGES")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nColumns:")
    print(list(df.columns))

    print(f"\nRows: {len(df):,}")

    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique():,}")
        print(f"  Missing values: {df[col].isna().sum():,}")

    # Duplicate edges
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate edges: {duplicates:,}")

    # Self-loops if this is a 2-column graph
    if len(df.columns) >= 2:
        source = df.columns[0]
        target = df.columns[1]
        self_loops = (df[source] == df[target]).sum()
        print(f"Self-loops: {self_loops:,}")


def inspect_classes(path):
    print("\n" + "=" * 70)
    print("TRANSACTION CLASSES")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nColumns:")
    print(list(df.columns))

    print(f"\nRows: {len(df):,}")

    print("\nFirst 10 rows:")
    print(df.head(10).to_string(index=False))

    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  Unique values: {df[col].nunique():,}")
        print(f"  Missing values: {df[col].isna().sum():,}")

        if df[col].nunique() < 20:
            print("  Value counts:")
            print(df[col].value_counts(dropna=False).to_string())


def inspect_features(path):
    print("\n" + "=" * 70)
    print("TRANSACTION FEATURES")
    print("=" * 70)

    # Header and first rows only
    sample = pd.read_csv(path, nrows=5)

    print(f"\nNumber of columns: {len(sample.columns)}")

    print("\nColumns:")
    for i, col in enumerate(sample.columns):
        print(f"  {i + 1:3}: {col}")

    print("\nFirst 5 rows:")
    print(sample.to_string(index=False))

    # Chunked audit
    total_rows = 0
    unique_ids = set()
    missing_values = 0

    id_column = sample.columns[0]

    for chunk in pd.read_csv(path, chunksize=100_000):
        total_rows += len(chunk)

        unique_ids.update(chunk[id_column].dropna().unique())

        missing_values += chunk.isna().sum().sum()

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Unique values in first column ({id_column}): {len(unique_ids):,}")
    print(f"Total missing cells: {missing_values:,}")


def main():
    inspect_file(
        RAW / "txs_features.csv",
        "TRANSACTION FEATURES"
    )

    inspect_edge_file(
        RAW / "txs_edgelist.csv"
    )

    inspect_classes(
        RAW / "txs_classes.csv"
    )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()