import pandas as pd
from collections import defaultdict, Counter

print("=" * 70)
print("AUDITING WALLET-TO-WALLET RELATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

print("\nLoading AddrTx...")
addr_tx = pd.read_csv(
    "data/raw/AddrTx_edgelist.csv",
    dtype=str
)

print(f"AddrTx rows: {len(addr_tx):,}")

print("\nLoading transaction-flow edges...")
tx_flow = pd.read_csv(
    "data/raw/txs_edgelist.csv",
    dtype=str
)

print(f"Money-flow edges: {len(tx_flow):,}")

print("\nColumns:")
print(f"AddrTx: {list(addr_tx.columns)}")
print(f"TxFlow: {list(tx_flow.columns)}")


# ------------------------------------------------------------
# 2. Build transaction -> participating wallets
# ------------------------------------------------------------

print("\nBuilding transaction -> wallet index...")

tx_to_wallets = defaultdict(set)

for row in addr_tx.itertuples(index=False):
    wallet = row.input_address
    tx = row.txId

    tx_to_wallets[tx].add(wallet)

print(f"Transactions with wallet data: {len(tx_to_wallets):,}")


# ------------------------------------------------------------
# 3. Construct wallet-to-wallet relationships
# ------------------------------------------------------------

print("\nConstructing wallet-to-wallet relationships...")

wallet_pairs = set()

# Keep some examples for inspection
examples = []

for row in tx_flow.itertuples(index=False):

    tx1 = row.txId1
    tx2 = row.txId2

    wallets_1 = tx_to_wallets.get(tx1, set())
    wallets_2 = tx_to_wallets.get(tx2, set())

    if not wallets_1 or not wallets_2:
        continue

    for wallet_a in wallets_1:
        for wallet_b in wallets_2:

            # Do not create self-relations
            if wallet_a == wallet_b:
                continue

            pair = (wallet_a, wallet_b)

            if pair not in wallet_pairs:
                wallet_pairs.add(pair)

                if len(examples) < 10:
                    examples.append(
                        (wallet_a, wallet_b, tx1, tx2)
                    )


# ------------------------------------------------------------
# 4. Basic statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nUnique wallet-to-wallet relationships: {len(wallet_pairs):,}")

wallets_in_relation = set()

for wallet_a, wallet_b in wallet_pairs:
    wallets_in_relation.add(wallet_a)
    wallets_in_relation.add(wallet_b)

print(f"Unique wallets involved: {len(wallets_in_relation):,}")


# ------------------------------------------------------------
# 5. Degree distribution
# ------------------------------------------------------------

out_degree = Counter()
in_degree = Counter()

for wallet_a, wallet_b in wallet_pairs:
    out_degree[wallet_a] += 1
    in_degree[wallet_b] += 1

print("\nWallet degree statistics:")

if out_degree:
    values = list(out_degree.values())

    print(f"Maximum outgoing degree: {max(values):,}")
    print(f"Average outgoing degree: {sum(values) / len(values):.2f}")

if in_degree:
    values = list(in_degree.values())

    print(f"Maximum incoming degree: {max(values):,}")
    print(f"Average incoming degree: {sum(values) / len(values):.2f}")


# ------------------------------------------------------------
# 6. Highly connected wallets
# ------------------------------------------------------------

print("\nTop 10 wallets by outgoing connections:")

for wallet, degree in out_degree.most_common(10):
    print(f"{wallet}: {degree:,}")


print("\nTop 10 wallets by incoming connections:")

for wallet, degree in in_degree.most_common(10):
    print(f"{wallet}: {degree:,}")


# ------------------------------------------------------------
# 7. Examples
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXAMPLES")
print("=" * 70)

for wallet_a, wallet_b, tx1, tx2 in examples:

    print(f"\n{wallet_a} -> transacts_with -> {wallet_b}")

    print("Evidence:")
    print(f"  {wallet_a} --participates_in--> {tx1}")
    print(f"  {tx1} --money_flows_to--> {tx2}")
    print(f"  {wallet_b} --participates_in--> {tx2}")


# ------------------------------------------------------------
# 8. Directionality check
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DIRECTIONALITY CHECK")
print("=" * 70)

reverse_pairs = 0

for wallet_a, wallet_b in wallet_pairs:
    if (wallet_b, wallet_a) in wallet_pairs:
        reverse_pairs += 1

print(f"Pairs with an observed reverse relationship: {reverse_pairs:,}")

if wallet_pairs:
    print(
        f"Percentage with reverse relationship: "
        f"{reverse_pairs / len(wallet_pairs) * 100:.2f}%"
    )


# ------------------------------------------------------------
# 9. Final interpretation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:\n"
    "These wallet-to-wallet relationships are DERIVED relationships.\n"
    "They should not yet be added to the KG or used for training.\n"
    "We are auditing whether they form a suitable link-prediction target."
)