from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "raw" / "wallets_features.csv"


def main():
    print("=" * 60)
    print("ELLIPTIC++ WALLET FEATURES INSPECTION")
    print("=" * 60)

    print(f"\nFile: {DATA_FILE}")
    print(f"Exists: {DATA_FILE.exists()}")

    if not DATA_FILE.exists():
        print("\nERROR: File was not found.")
        return

    size_mb = DATA_FILE.stat().st_size / (1024 * 1024)
    print(f"Size: {size_mb:.2f} MB")

    print("\nHeader and first 10 rows:")
    print("-" * 60)

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.reader(file)

        for row_number, row in enumerate(reader):
            print(f"Row {row_number}: {row}")

            if row_number >= 10:
                break

    print("\n" + "=" * 60)
    print("INSPECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()