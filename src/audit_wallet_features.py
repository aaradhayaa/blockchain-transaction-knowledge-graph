from pathlib import Path
import pandas as pd

FILE = Path("data/raw/wallets_features.csv")
OUTPUT = Path("data/processed/wallet_features_audit.txt")


def main():

    print("=" * 70)
    print("ELLIPTIC++ WALLET FEATURES AUDIT")
    print("=" * 70)

    print("\nLoading wallet features...")
    df = pd.read_csv(FILE)

    unique_wallets = df["address"].nunique()
    unique_steps = sorted(df["Time step"].unique())

    rows_per_wallet = df.groupby("address").size()
    rows_per_step = df.groupby("Time step").size()

    duplicate_wallet_time = df.duplicated(
        subset=["address", "Time step"]
    ).sum()

    # Feature variation
    feature_columns = [
        c for c in df.columns
        if c not in ["address", "Time step"]
    ]

    variation = []

    for col in feature_columns:
        distinct = df.groupby("address")[col].nunique(
            dropna=False
        )

        changed = (distinct > 1).sum()
        total = len(distinct)

        variation.append(
            (col, changed, 100 * changed / total)
        )

    variation.sort(key=lambda x: x[2], reverse=True)

    # Missing values
    missing = df.isna().sum()
    missing = missing[missing > 0]

    # Create report
    lines = []

    lines.append("ELLIPTIC++ WALLET FEATURES AUDIT")
    lines.append("=" * 70)

    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns: {len(df.columns):,}")
    lines.append(f"Unique wallets: {unique_wallets:,}")
    lines.append(f"Unique time steps: {len(unique_steps)}")
    lines.append(f"Time steps: {unique_steps}")

    lines.append("")
    lines.append("ROWS PER WALLET")
    lines.append("-" * 70)
    lines.append(f"Minimum: {rows_per_wallet.min():,}")
    lines.append(f"Maximum: {rows_per_wallet.max():,}")
    lines.append(f"Mean: {rows_per_wallet.mean():.2f}")
    lines.append(f"Median: {rows_per_wallet.median():.2f}")

    lines.append("")
    lines.append("TOP ROW-COUNT DISTRIBUTION")
    lines.append("-" * 70)

    distribution = rows_per_wallet.value_counts().sort_index()

    for n, count in distribution.head(20).items():
        lines.append(
            f"{n:>4} observations: {count:>10,} wallets"
        )

    lines.append("")
    lines.append("ROWS PER TIME STEP")
    lines.append("-" * 70)

    for step, count in rows_per_step.items():
        lines.append(
            f"Time step {step:>2}: {count:>10,} rows"
        )

    lines.append("")
    lines.append("DUPLICATE WALLET-TIME PAIRS")
    lines.append("-" * 70)
    lines.append(
        f"Duplicate (address, Time step) rows: "
        f"{duplicate_wallet_time:,}"
    )

    lines.append("")
    lines.append("FEATURE VARIATION")
    lines.append("-" * 70)
    lines.append(
        "Percentage of wallets for which the feature "
        "takes more than one value:"
    )

    for col, changed, percentage in variation:
        lines.append(
            f"{col:<45} "
            f"{changed:>10,} wallets "
            f"({percentage:>6.2f}%)"
        )

    lines.append("")
    lines.append("MISSING VALUES")
    lines.append("-" * 70)

    if missing.empty:
        lines.append("No missing values.")
    else:
        for col, count in missing.items():
            lines.append(f"{col:<45} {count:>10,}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nAudit complete.")
    print(f"Results saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()