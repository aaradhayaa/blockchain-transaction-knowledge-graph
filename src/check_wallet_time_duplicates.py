from pathlib import Path
import pandas as pd

FILE = Path("data/raw/wallets_features.csv")

df = pd.read_csv(FILE)

feature_columns = [
    c for c in df.columns
    if c not in ["address", "Time step"]
]

# Keep only wallet-time groups that occur more than once
dup = df[df.duplicated(
    subset=["address", "Time step"],
    keep=False
)].copy()

group_sizes = (
    dup.groupby(["address", "Time step"])
    .size()
)

# For each duplicated wallet-time group,
# count how many distinct values occur in each feature.
distinct_per_feature = (
    dup.groupby(["address", "Time step"])[feature_columns]
    .nunique(dropna=False)
)

# A duplicated group is "identical" if every feature
# has exactly one distinct value.
identical_groups = (
    distinct_per_feature.eq(1).all(axis=1)
)

conflicting_groups = ~identical_groups

print("=" * 70)
print("DUPLICATE WALLET-TIME CHECK")
print("=" * 70)

print(f"Duplicate rows: {len(dup):,}")
print(f"Duplicate wallet-time groups: {len(group_sizes):,}")

print("\nGROUP SIZE")
print("-" * 70)
print(f"Minimum rows in duplicated group: {group_sizes.min():,}")
print(f"Maximum rows in duplicated group: {group_sizes.max():,}")
print(f"Mean rows per duplicated group: {group_sizes.mean():.2f}")

print("\nFEATURE CONSISTENCY")
print("-" * 70)
print(
    f"Groups with identical feature values: "
    f"{identical_groups.sum():,}"
)
print(
    f"Groups with conflicting feature values: "
    f"{conflicting_groups.sum():,}"
)

print("\nPERCENTAGES")
print("-" * 70)

total_groups = len(group_sizes)

print(
    f"Identical: "
    f"{100 * identical_groups.sum() / total_groups:.2f}%"
)

print(
    f"Conflicting: "
    f"{100 * conflicting_groups.sum() / total_groups:.2f}%"
)

# Show a few conflicting examples if they exist
if conflicting_groups.any():

    print("\nEXAMPLE CONFLICTING GROUPS")
    print("-" * 70)

    conflict_keys = conflicting_groups[
        conflicting_groups
    ].index[:5]

    for address, timestep in conflict_keys:

        example = df[
            (df["address"] == address) &
            (df["Time step"] == timestep)
        ]

        print(
            f"\nAddress: {address}"
            f"\nTime step: {timestep}"
        )

        # Only display features that actually differ
        differing = []

        for col in feature_columns:
            if example[col].nunique(dropna=False) > 1:
                differing.append(col)

        print("Differing features:")
        for col in differing:
            print(f"  - {col}")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)