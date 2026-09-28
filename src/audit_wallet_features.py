from pathlib import Path
import pandas as pd


FILE = Path("data/raw/wallets_features.csv")


def main():
    print("=" * 70)
    print("ELLIPTIC++ WALLET FEATURES AUDIT")
    print("=" * 70)

    print("\nLoading wallet features...")
    df = pd.read_csv(FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # ---------------------------------------------------------
    # 1. Wallet and time structure
    # ---------------------------------------------------------
    print("\n[1] WALLET / TIME STRUCTURE")
    print("-" * 70)

    unique_wallets = df["address"].nunique()
    unique_steps = sorted(df["Time step"].unique())

    print(f"Unique wallets: {unique_wallets:,}")
    print(f"Unique time steps: {len(unique_steps)}")
    print(f"Time steps: {unique_steps}")

    # ---------------------------------------------------------
    # 2. Rows per wallet
    # ---------------------------------------------------------
    print("\n[2] ROWS PER WALLET")
    print("-" * 70)

    rows_per_wallet = df.groupby("address").size()

    print(f"Minimum: {rows_per_wallet.min():,}")
    print(f"Maximum: {rows_per_wallet.max():,}")
    print(f"Mean:    {rows_per_wallet.mean():.2f}")
    print(f"Median:  {rows_per_wallet.median():.2f}")

    print("\nNumber of wallets by number of time-step observations:")

    distribution = rows_per_wallet.value_counts().sort_index()

    for n_rows, count in distribution.items():
        print(f"  {n_rows:>3} observations: {count:>10,} wallets")

    # ---------------------------------------------------------
    # 3. Rows per time step
    # ---------------------------------------------------------
    print("\n[3] ROWS PER TIME STEP")
    print("-" * 70)

    rows_per_step = df.groupby("Time step").size()

    print(
        f"Minimum rows in a time step: {rows_per_step.min():,}"
    )
    print(
        f"Maximum rows in a time step: {rows_per_step.max():,}"
    )
    print(
        f"Mean rows per time step:      {rows_per_step.mean():,.0f}"
    )

    print("\nFirst 10 time steps:")
    print(rows_per_step.head(10).to_string())

    print("\nLast 10 time steps:")
    print(rows_per_step.tail(10).to_string())

    # ---------------------------------------------------------
    # 4. Duplicate wallet/time observations
    # ---------------------------------------------------------
    print("\n[4] DUPLICATE WALLET-TIME PAIRS")
    print("-" * 70)

    duplicates = df.duplicated(
        subset=["address", "Time step"]
    ).sum()

    print(
        f"Duplicate (address, Time step) rows: {duplicates:,}"
    )

    # ---------------------------------------------------------
    # 5. Feature variation
    # ---------------------------------------------------------
    print("\n[5] FEATURE VARIATION OVER TIME")
    print("-" * 70)

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
            (
                col,
                changed,
                100 * changed / total
            )
        )

    variation.sort(key=lambda x: x[2], reverse=True)

    print("\nMost time-varying features:")
    for col, changed, percentage in variation[:10]:
        print(
            f"  {col:<40} "
            f"{changed:>8,} wallets "
            f"({percentage:>6.2f}%)"
        )

    print("\nLeast time-varying features:")
    for col, changed, percentage in variation[-10:]:
        print(
            f"  {col:<40} "
            f"{changed:>8,} wallets "
            f"({percentage:>6.2f}%)"
        )

    # ---------------------------------------------------------
    # 6. Missing values
    # ---------------------------------------------------------
    print("\n[6] MISSING VALUES")
    print("-" * 70)

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.sort_values(ascending=False).to_string())

    # ---------------------------------------------------------
    # 7. Example repeated wallet
    # ---------------------------------------------------------
    print("\n[7] EXAMPLE REPEATED WALLET")
    print("-" * 70)

    example_wallet = rows_per_wallet.idxmax()

    subset = df[df["address"] == example_wallet].copy()

    columns = [
        "address",
        "Time step",
        "total_txs",
        "btc_transacted_total",
        "btc_sent_total",
        "btc_received_total",
        "fees_total",
        "num_addr_transacted_multiple",
    ]

    columns = [c for c in columns if c in df.columns]

    print(subset[columns].sort_values("Time step").to_string(index=False))

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()