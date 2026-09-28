from pathlib import Path

import pandas as pd
from tqdm import tqdm


# --------------------------------------------------
# 1. Set project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Helper function to create triples
# --------------------------------------------------

def create_triples(
    df,
    head_column,
    relation,
    tail_column,
):
    """
    Convert an edge dataframe into KG triples:
    (head, relation, tail)
    """

    triples = pd.DataFrame(
        {
            "head": df[head_column].astype(str),
            "relation": relation,
            "tail": df[tail_column].astype(str),
        }
    )

    # Remove duplicate relationships
    triples = triples.drop_duplicates()

    # Remove self-loops
    triples = triples[triples["head"] != triples["tail"]]

    return triples


# --------------------------------------------------
# 3. Load wallet-to-wallet relationships
# --------------------------------------------------

print("Loading wallet-to-wallet edges...")

addr_addr = pd.read_csv(
    RAW_DIR / "AddrAddr_edgelist.csv",
    dtype=str,
)

wallet_triples = create_triples(
    addr_addr,
    head_column="input_address",
    relation="transacted_with",
    tail_column="output_address",
)

print(f"Wallet-to-wallet triples: {len(wallet_triples):,}")


# --------------------------------------------------
# 4. Load wallet-to-transaction relationships
# --------------------------------------------------

print("Loading wallet-to-transaction edges...")

addr_tx = pd.read_csv(
    RAW_DIR / "AddrTx_edgelist.csv",
    dtype=str,
)

wallet_transaction_triples = create_triples(
    addr_tx,
    head_column="input_address",
    relation="participates_in",
    tail_column="txId",
)

print(
    "Wallet-to-transaction triples: "
    f"{len(wallet_transaction_triples):,}"
)


# --------------------------------------------------
# 5. Load transaction-to-wallet relationships
# --------------------------------------------------

print("Loading transaction-to-wallet edges...")

tx_addr = pd.read_csv(
    RAW_DIR / "TxAddr_edgelist.csv",
    dtype=str,
)

transaction_wallet_triples = create_triples(
    tx_addr,
    head_column="txId",
    relation="has_output",
    tail_column="output_address",
)

print(
    "Transaction-to-wallet triples: "
    f"{len(transaction_wallet_triples):,}"
)


# --------------------------------------------------
# 6. Combine all relationships
# --------------------------------------------------

print("\nCombining triples...")

all_triples = pd.concat(
    [
        wallet_triples,
        wallet_transaction_triples,
        transaction_wallet_triples,
    ],
    ignore_index=True,
)

# Remove duplicate triples across the combined dataset
all_triples = all_triples.drop_duplicates()

# Save the complete KG triple dataset
output_path = PROCESSED_DIR / "bitcoin_triples_full.csv"

all_triples.to_csv(output_path, index=False)


# --------------------------------------------------
# 7. Print summary
# --------------------------------------------------

print("\nKnowledge graph triple dataset created!")

print(f"Total triples: {len(all_triples):,}")

print("\nTriples by relationship:")
print(all_triples["relation"].value_counts())

print(f"\nSaved to: {output_path}")