from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

KG_FILE = Path("data/processed/knowledge_graph_triples.csv")

OUTPUT_DIR = Path("results/figures")
OUTPUT_PNG = OUTPUT_DIR / "knowledge_graph.png"
OUTPUT_PDF = OUTPUT_DIR / "knowledge_graph.pdf"

# Smaller, more readable representative graph
N_TRANSACTIONS = 18
MAX_WALLETS = 45

# Avoid very highly connected hubs
MIN_TX_DEGREE = 2
MAX_TX_DEGREE = 10

SEED = 42


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

print("=" * 70)
print("CREATING REPORT-QUALITY KNOWLEDGE GRAPH FIGURE")
print("=" * 70)

kg = pd.read_csv(
    KG_FILE,
    dtype={
        "head": "string",
        "relation": "string",
        "tail": "string",
    }
)

print(f"Total triples: {len(kg):,}")
print(f"Relations: {kg['relation'].nunique():,}")

entities = pd.unique(
    kg[["head", "tail"]].values.ravel()
)

print(f"Unique entities: {len(entities):,}")


# ============================================================
# SEPARATE RELATIONS
# ============================================================

participates = kg[
    kg["relation"] == "participates_in"
]

has_output = kg[
    kg["relation"] == "has_output"
]

money_flow = kg[
    kg["relation"] == "money_flows_to"
]


# ============================================================
# IDENTIFY TRANSACTIONS
# ============================================================

transaction_ids = set()

transaction_ids.update(
    participates["tail"].dropna()
)

transaction_ids.update(
    has_output["head"].dropna()
)

transaction_ids.update(
    money_flow["head"].dropna()
)

transaction_ids.update(
    money_flow["tail"].dropna()
)

print(
    f"Transaction entities: "
    f"{len(transaction_ids):,}"
)


# ============================================================
# BUILD TRANSACTION-TO-TRANSACTION GRAPH
# ============================================================

tx_graph = nx.Graph()

tx_graph.add_edges_from(
    zip(
        money_flow["head"],
        money_flow["tail"]
    )
)

tx_graph.remove_nodes_from(
    [
        node
        for node in tx_graph.nodes
        if node not in transaction_ids
    ]
)


# ============================================================
# FIND MODERATELY CONNECTED TRANSACTIONS
# ============================================================

print("\nSelecting moderately connected transactions...")

tx_degrees = dict(
    tx_graph.degree()
)

moderate_transactions = {
    node
    for node, degree in tx_degrees.items()
    if MIN_TX_DEGREE <= degree <= MAX_TX_DEGREE
}

print(
    f"Transactions with degree "
    f"{MIN_TX_DEGREE}-{MAX_TX_DEGREE}: "
    f"{len(moderate_transactions):,}"
)


# ============================================================
# FIND CONNECTED MODERATE-DEGREE COMPONENT
# ============================================================

moderate_graph = tx_graph.subgraph(
    moderate_transactions
).copy()

components = list(
    nx.connected_components(moderate_graph)
)

components = sorted(
    components,
    key=len,
    reverse=True
)

if not components:

    raise RuntimeError(
        "No suitable moderate-degree transaction "
        "subgraph was found."
    )


largest_component = components[0]

print(
    f"Largest moderate-degree component: "
    f"{len(largest_component):,} transactions"
)


# ============================================================
# SELECT TRANSACTIONS FROM THIS COMPONENT
# ============================================================

component_graph = moderate_graph.subgraph(
    largest_component
).copy()

# Choose a transaction near the middle of the degree range.
# This avoids an extreme hub.
candidate_transactions = sorted(
    component_graph.nodes,
    key=lambda node: (
        abs(
            tx_degrees[node]
            - (
                MIN_TX_DEGREE
                + MAX_TX_DEGREE
            ) / 2
        ),
        str(node)
    )
)

seed_transaction = candidate_transactions[0]

print(
    f"Seed transaction: "
    f"{seed_transaction}"
)

print(
    f"Seed transaction degree: "
    f"{tx_degrees[seed_transaction]}"
)


# ============================================================
# BFS THROUGH MODERATE-DEGREE SUBGRAPH
# ============================================================

selected_transactions = []

visited = {
    seed_transaction
}

queue = [
    seed_transaction
]

while queue and len(selected_transactions) < N_TRANSACTIONS:

    current = queue.pop(0)

    selected_transactions.append(
        current
    )

    neighbours = sorted(
        component_graph.neighbors(current),
        key=lambda node: (
            abs(
                tx_degrees[node]
                - (
                    MIN_TX_DEGREE
                    + MAX_TX_DEGREE
                ) / 2
            ),
            str(node)
        )
    )

    for neighbour in neighbours:

        if neighbour not in visited:

            visited.add(neighbour)
            queue.append(neighbour)


selected_transactions = set(
    selected_transactions[:N_TRANSACTIONS]
)

print(
    f"Selected transactions: "
    f"{len(selected_transactions):,}"
)


# ============================================================
# GET RELATIONSHIPS FOR SELECTED TRANSACTIONS
# ============================================================

selected_participates = participates[
    participates["tail"].isin(
        selected_transactions
    )
]

selected_outputs = has_output[
    has_output["head"].isin(
        selected_transactions
    )
]

selected_money_flow = money_flow[
    money_flow["head"].isin(
        selected_transactions
    )
    &
    money_flow["tail"].isin(
        selected_transactions
    )
]


print(
    f"Participation relationships: "
    f"{len(selected_participates):,}"
)

print(
    f"Output relationships: "
    f"{len(selected_outputs):,}"
)

print(
    f"Transaction-flow relationships: "
    f"{len(selected_money_flow):,}"
)


# ============================================================
# IDENTIFY WALLET ENDPOINTS CORRECTLY
# ============================================================

# participates_in:
#
#     WALLET -> TRANSACTION
#
# has_output:
#
#     TRANSACTION -> WALLET
#
# Therefore:
#
#     wallet = participates.head
#     wallet = has_output.tail

wallet_endpoints = pd.concat(
    [
        selected_participates["head"],
        selected_outputs["tail"]
    ],
    ignore_index=True
)

wallet_degree = (
    wallet_endpoints
    .value_counts()
)


# ============================================================
# AVOID HIGH-DEGREE WALLET HUBS
# ============================================================

# Prefer wallets involved in only a few relationships.
# This prevents one wallet from producing a giant bundle
# of overlapping lines.

moderate_wallets = wallet_degree[
    wallet_degree <= 5
]

if len(moderate_wallets) >= MAX_WALLETS:

    selected_wallets = set(
        moderate_wallets
        .sort_index()
        .head(MAX_WALLETS)
        .index
    )

else:

    # If fewer than MAX_WALLETS satisfy the threshold,
    # take the least-connected wallets available.
    selected_wallets = set(
        wallet_degree
        .sort_values()
        .head(MAX_WALLETS)
        .index
    )


print(
    f"Selected wallets: "
    f"{len(selected_wallets):,}"
)


# ============================================================
# FILTER FINAL RELATIONSHIPS
# ============================================================

final_participates = selected_participates[
    selected_participates["head"].isin(
        selected_wallets
    )
]

final_outputs = selected_outputs[
    selected_outputs["tail"].isin(
        selected_wallets
    )
]

final_money_flow = selected_money_flow


# ============================================================
# BUILD GRAPH
# ============================================================

G = nx.MultiDiGraph()


# ------------------------------------------------------------
# Transaction nodes
# ------------------------------------------------------------

for tx in selected_transactions:

    G.add_node(
        str(tx),
        node_type="transaction"
    )


# ------------------------------------------------------------
# Wallet nodes
# ------------------------------------------------------------

for wallet in selected_wallets:

    G.add_node(
        str(wallet),
        node_type="wallet"
    )


# ------------------------------------------------------------
# participates_in
# ------------------------------------------------------------

for row in final_participates.itertuples(
    index=False
):

    G.add_edge(
        str(row.head),
        str(row.tail),
        relation="participates_in"
    )


# ------------------------------------------------------------
# has_output
# ------------------------------------------------------------

for row in final_outputs.itertuples(
    index=False
):

    G.add_edge(
        str(row.head),
        str(row.tail),
        relation="has_output"
    )


# ------------------------------------------------------------
# money_flows_to
# ------------------------------------------------------------

for row in final_money_flow.itertuples(
    index=False
):

    G.add_edge(
        str(row.head),
        str(row.tail),
        relation="money_flows_to"
    )


print(
    f"\nFigure nodes: "
    f"{G.number_of_nodes():,}"
)

print(
    f"Figure edges: "
    f"{G.number_of_edges():,}"
)


# ============================================================
# REMOVE ISOLATED NODES
# ============================================================

isolated_nodes = list(
    nx.isolates(G)
)

if isolated_nodes:

    G.remove_nodes_from(
        isolated_nodes
    )

    print(
        f"Removed isolated nodes: "
        f"{len(isolated_nodes):,}"
    )


# ============================================================
# CREATE SIMPLE GRAPH FOR LAYOUT
# ============================================================

# MultiDiGraph contains relation-specific edges.
# For layout purposes we only need to know which entities
# are connected at all.

layout_graph = nx.Graph()

layout_graph.add_nodes_from(
    G.nodes()
)

layout_graph.add_edges_from(
    [
        (u, v)
        for u, v, data
        in G.edges(data=True)
    ]
)


# ============================================================
# CALCULATE FORCE-DIRECTED LAYOUT
# ============================================================

print("\nCalculating network layout...")

positions = nx.spring_layout(
    layout_graph,
    seed=SEED,
    k=2.0,
    iterations=200,
    scale=1.0
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(16, 12)
)


# ============================================================
# DRAW EDGES
# ============================================================

# participates_in
participates_edges = [
    (u, v)
    for u, v, data
    in G.edges(data=True)
    if data["relation"] == "participates_in"
]

nx.draw_networkx_edges(
    G,
    positions,
    edgelist=participates_edges,
    ax=ax,
    edge_color="black",
    style="solid",
    width=1.0,
    alpha=0.45,
    arrows=True,
    arrowsize=10,
    connectionstyle="arc3,rad=0.03"
)


# has_output
output_edges = [
    (u, v)
    for u, v, data
    in G.edges(data=True)
    if data["relation"] == "has_output"
]

nx.draw_networkx_edges(
    G,
    positions,
    edgelist=output_edges,
    ax=ax,
    edge_color="black",
    style="dashed",
    width=1.1,
    alpha=0.50,
    arrows=True,
    arrowsize=10,
    connectionstyle="arc3,rad=0.03"
)


# money_flows_to
flow_edges = [
    (u, v)
    for u, v, data
    in G.edges(data=True)
    if data["relation"] == "money_flows_to"
]

nx.draw_networkx_edges(
    G,
    positions,
    edgelist=flow_edges,
    ax=ax,
    edge_color="black",
    style="dotted",
    width=2.0,
    alpha=0.75,
    arrows=True,
    arrowsize=13,
    connectionstyle="arc3,rad=0.10"
)


# ============================================================
# GET NODE TYPES
# ============================================================

transaction_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data["node_type"] == "transaction"
]

wallet_nodes = [
    node
    for node, data
    in G.nodes(data=True)
    if data["node_type"] == "wallet"
]


# ============================================================
# DRAW WALLETS FIRST
# ============================================================

nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=wallet_nodes,
    node_shape="o",
    node_size=400,
    node_color="#76A5D5",
    edgecolors="black",
    linewidths=1.0,
    ax=ax
)


# ============================================================
# DRAW TRANSACTIONS LAST
# ============================================================

# Drawing transactions after wallets makes the squares
# visually dominant and prevents them from being hidden.

nx.draw_networkx_nodes(
    G,
    positions,
    nodelist=transaction_nodes,
    node_shape="s",
    node_size=2200,
    node_color="#F4B942",
    edgecolors="black",
    linewidths=2.5,
    ax=ax
)


# ============================================================
# TRANSACTION LABELS
# ============================================================

transaction_labels = {}

for tx in transaction_nodes:

    tx_string = str(tx)

    transaction_labels[tx] = (
        "TX\n"
        + tx_string[-6:]
    )


nx.draw_networkx_labels(
    G,
    positions,
    labels=transaction_labels,
    font_size=8,
    font_weight="bold",
    font_color="black",
    ax=ax
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="s",
        color="white",
        markerfacecolor="#F4B942",
        markeredgecolor="black",
        markersize=13,
        label="Transaction"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="white",
        markerfacecolor="#76A5D5",
        markeredgecolor="black",
        markersize=10,
        label="Wallet"
    ),

    Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=1.2,
        label="participates_in"
    ),

    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="has_output"
    ),

    Line2D(
        [0],
        [0],
        color="black",
        linestyle=":",
        linewidth=2.0,
        label="money_flows_to"
    )
]


ax.legend(
    handles=legend_elements,
    loc="upper left",
    fontsize=10,
    title="Knowledge graph elements",
    frameon=True
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Representative Subgraph of the Bitcoin Transaction Knowledge Graph",
    fontsize=19,
    fontweight="bold",
    pad=20
)

ax.text(
    0.5,
    0.015,
    (
        f"Representative sample: "
        f"{len(transaction_nodes)} transactions, "
        f"{len(wallet_nodes)} wallets, "
        f"{G.number_of_edges()} relationships"
    ),
    transform=ax.transAxes,
    ha="center",
    fontsize=10
)


# ============================================================
# CLEAN UP
# ============================================================

ax.set_axis_off()

plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REPORT FIGURE CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"Transactions shown: "
    f"{len(transaction_nodes):,}"
)

print(
    f"Wallets shown: "
    f"{len(wallet_nodes):,}"
)

print(
    f"Total nodes: "
    f"{G.number_of_nodes():,}"
)

print(
    f"Relationships shown: "
    f"{G.number_of_edges():,}"
)

print("\nPNG:")
print(OUTPUT_PNG)

print("\nPDF:")
print(OUTPUT_PDF)

print("=" * 70)