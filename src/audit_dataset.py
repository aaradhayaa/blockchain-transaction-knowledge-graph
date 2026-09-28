from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")


def inspect_file(filename):
    path = RAW_DIR / filename

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    if not path.exists():
        print("NOT FOUND")
        return

    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

    df = pd.read_csv(path, nrows=5)

    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")

    print("\nNumber of columns:", len(df.columns))

    print("\nFirst 5 rows:")
    print(df.to_string(index=False))

    print("\nDtypes:")
    print(df.dtypes.to_string())


def main():
    print("ELLIPTIC++ DATASET AUDIT")

    files = [
        "AddrAddr_edgelist.csv",
        "AddrTx_edgelist.csv",
        "TxAddr_edgelist.csv",
        "wallets_features.csv",
    ]

    for filename in files:
        inspect_file(filename)


if __name__ == "__main__":
    main()