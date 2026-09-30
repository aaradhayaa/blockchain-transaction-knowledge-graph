from pathlib import Path

import pandas as pd


# ============================================================
# FILES
# ============================================================

DATA_DIR = Path("data/raw")

TXADDR_FILE = DATA_DIR / "TxAddr_edgelist.csv"
ADDRTX_FILE = DATA_DIR / "AddrTx_edgelist.csv"
TXFLOW_FILE = DATA_DIR / "txs_edgelist.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("AUDITING MONEY-FLOW RECONSTRUCTABILITY")
print("=" * 70)


print("\nLoading TxAddr...")
txaddr = pd.read_csv(
    TXADDR_FILE,
    dtype="string"
)

print(
    f"TxAddr rows: {len(txaddr):,}"
)

print("\nLoading AddrTx...")
addrtx = pd.read_csv(
    ADDRTX_FILE,
    dtype="string"
)

print(
    f"AddrTx rows: {len(addrtx):,}"
)

print("\nLoading transaction-flow edges...")
txflow = pd.read_csv(
    TXFLOW_FILE,
    dtype="string"
)

print(
    f"Money-flow edges: {len(txflow):,}"
)


# ============================================================
# CHECK COLUMN NAMES
# ============================================================

print("\nColumns:")

print(
    "TxAddr:",
    list(txaddr.columns)
)

print(
    "AddrTx:",
    list(addrtx.columns)
)

print(
    "TxFlow:",
    list(txflow.columns)
)


# ============================================================
# BUILD TRANSACTION -> OUTPUT WALLETS
# ============================================================

print("\nBuilding transaction -> output-wallet index...")

tx_to_output_wallets = (
    txaddr
    .groupby("txId")["output_address"]
    .agg(set)
    .to_dict()
)

print(
    f"Transactions with output wallets: "
    f"{len(tx_to_output_wallets):,}"
)


# ============================================================
# BUILD TRANSACTION -> INPUT WALLETS
# ============================================================

print("Building transaction -> input-wallet index...")

tx_to_input_wallets = (
    addrtx
    .groupby("txId")["input_address"]
    .agg(set)
    .to_dict()
)

print(
    f"Transactions with input wallets: "
    f"{len(tx_to_input_wallets):,}"
)


# ============================================================
# AUDIT MONEY-FLOW EDGES
# ============================================================

print("\nChecking money-flow edges...")

reconstructable_count = 0
not_reconstructable_count = 0

missing_source_count = 0
missing_target_count = 0

examples_reconstructable = []
examples_not_reconstructable = []


for row in txflow.itertuples(index=False):

    source_tx = row.txId1
    target_tx = row.txId2

    source_wallets = tx_to_output_wallets.get(
        source_tx,
        set()
    )

    target_wallets = tx_to_input_wallets.get(
        target_tx,
        set()
    )

    if not source_wallets:
        missing_source_count += 1

    if not target_wallets:
        missing_target_count += 1

    shared_wallets = (
        source_wallets
        & target_wallets
    )

    if shared_wallets:

        reconstructable_count += 1

        if len(examples_reconstructable) < 10:

            examples_reconstructable.append(
                (
                    source_tx,
                    target_tx,
                    list(shared_wallets)[:5]
                )
            )

    else:

        not_reconstructable_count += 1

        if len(examples_not_reconstructable) < 10:

            examples_not_reconstructable.append(
                (
                    source_tx,
                    target_tx
                )
            )


# ============================================================
# RESULTS
# ============================================================

total = len(txflow)

reconstructable_pct = (
    reconstructable_count / total * 100
)

not_reconstructable_pct = (
    not_reconstructable_count / total * 100
)


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Total money-flow edges: "
    f"{total:,}"
)

print(
    f"\nReconstructable through a shared wallet: "
    f"{reconstructable_count:,}"
)

print(
    f"Percentage reconstructable: "
    f"{reconstructable_pct:.2f}%"
)

print(
    f"\nNOT reconstructable through a shared wallet: "
    f"{not_reconstructable_count:,}"
)

print(
    f"Percentage NOT reconstructable: "
    f"{not_reconstructable_pct:.2f}%"
)

print(
    f"\nSource transactions without output-wallet data: "
    f"{missing_source_count:,}"
)

print(
    f"Target transactions without input-wallet data: "
    f"{missing_target_count:,}"
)


# ============================================================
# RECONSTRUCTABLE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLES: RECONSTRUCTABLE")
print("=" * 70)

for source_tx, target_tx, wallets in examples_reconstructable:

    print(
        f"\n{source_tx} -> {target_tx}"
    )

    print(
        "Shared wallet(s):",
        wallets
    )

    print(
        f"Therefore:\n"
        f"  {source_tx}"
        f" --has_output--> "
        f"{wallets[0]}"
    )

    print(
        f"  {wallets[0]}"
        f" --participates_in--> "
        f"{target_tx}"
    )


# ============================================================
# NON-RECONSTRUCTABLE EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("EXAMPLES: NOT RECONSTRUCTABLE")
print("=" * 70)

for source_tx, target_tx in examples_not_reconstructable:

    print(
        f"{source_tx} -> {target_tx}"
    )


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if reconstructable_pct >= 90:

    print(
        "\nWARNING:"
    )

    print(
        "The majority of money-flow relationships can be "
        "reconstructed from wallet relationships."
    )

    print(
        "Using money_flows_to as the prediction target "
        "would therefore carry substantial structural leakage risk."
    )

elif reconstructable_pct >= 50:

    print(
        "\nCAUTION:"
    )

    print(
        "A substantial portion of money-flow relationships "
        "can be reconstructed from wallet relationships."
    )

    print(
        "The prediction task requires a more careful split "
        "and possibly a restricted KG representation."
    )

else:

    print(
        "\nRESULT:"
    )

    print(
        "Most money-flow relationships cannot be directly "
        "reconstructed through a shared wallet."
    )

    print(
        "This makes money_flows_to a potentially viable "
        "link-prediction target, subject to further leakage checks."
    )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)