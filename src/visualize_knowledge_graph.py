from pathlib import Path
import pandas as pd
import networkx as nx
from pyvis.network import Network


# ============================================================
# CONFIGURATION
# ============================================================

KG_FILE = Path("data/processed/knowledge_graph_triples.csv")
OUTPUT_DIR = Path("results/kg_visualization")
OUTPUT_HTML = OUTPUT_DIR / "index.html"

# Size of the visualisation
N_TRANSACTIONS = 40
MAX_WALLETS = 150

# Reproducibility
SEED = 42


# ============================================================
# LOAD KNOWLEDGE GRAPH
# ============================================================

print("=" * 70)
print("LOADING KNOWLEDGE GRAPH")
print("=" * 70)

# IMPORTANT:
# Read entity IDs as strings.
# Wallet addresses and transaction IDs must not be converted
# into different numeric/string types.
kg = pd.read_csv(
    KG_FILE,
    dtype={
        "head": "string",
        "relation": "string",
        "tail": "string",
    }
)

print(f"Total triples in KG: {len(kg):,}")
print(f"Relations: {kg['relation'].nunique():,}")
print(f"Unique entities: {pd.unique(kg[['head', 'tail']].values.ravel()).size:,}")

print("\nTriples by relation:")
print(kg["relation"].value_counts())


# ============================================================
# IDENTIFY TRANSACTIONS
# ============================================================

# A transaction can appear:
#   - as the tail of participates_in
#   - as the head of has_output
#   - as either head or tail of money_flows_to

participates = kg[kg["relation"] == "participates_in"]

has_output = kg[kg["relation"] == "has_output"]

money_flow = kg[kg["relation"] == "money_flows_to"]


transaction_ids = set()

transaction_ids.update(participates["tail"].dropna())
transaction_ids.update(has_output["head"].dropna())
transaction_ids.update(money_flow["head"].dropna())
transaction_ids.update(money_flow["tail"].dropna())

print(f"\nTransaction entities identified: {len(transaction_ids):,}")


# ============================================================
# SELECT A CONNECTED TRANSACTION SUBGRAPH
# ============================================================

print("\nSelecting transactions for visualization...")

# Build ONLY the transaction-to-transaction graph.
#
# This graph is used to select a connected group of transactions.
# We do NOT later expand recursively through this graph.

tx_graph = nx.Graph()

tx_edges = list(
    zip(
        money_flow["head"],
        money_flow["tail"]
    )
)

tx_graph.add_edges_from(tx_edges)

# Remove anything that somehow isn't identified as a transaction.
tx_graph.remove_nodes_from(
    [node for node in tx_graph.nodes if node not in transaction_ids]
)

print(f"Transaction graph nodes: {tx_graph.number_of_nodes():,}")
print(f"Transaction graph edges: {tx_graph.number_of_edges():,}")


# ------------------------------------------------------------
# Find a well-connected starting transaction
# ------------------------------------------------------------

# We use degree to find a transaction that actually participates
# in several transaction-to-transaction relationships.

degree_sorted = sorted(
    tx_graph.degree(),
    key=lambda x: x[1],
    reverse=True
)

# Avoid selecting an extremely high-degree hub.
# Instead, look through the top candidates and choose one that
# can produce a useful local neighbourhood.
seed_transaction = None

for node, degree in degree_sorted[:500]:
    if degree >= 2:
        seed_transaction = node
        break

if seed_transaction is None:
    raise RuntimeError(
        "Could not find a suitable transaction seed."
    )

print(f"Seed transaction selected: {seed_transaction}")


# ------------------------------------------------------------
# Breadth-first traversal
# ------------------------------------------------------------

# IMPORTANT:
# This traversal stops once we have N_TRANSACTIONS.
#
# It does NOT add every transaction connected to the selected
# transactions.

selected_transactions = []
visited = set()

queue = [seed_transaction]
visited.add(seed_transaction)

while queue and len(selected_transactions) < N_TRANSACTIONS:

    current = queue.pop(0)

    selected_transactions.append(current)

    neighbours = list(tx_graph.neighbors(current))

    # Deterministic ordering
    neighbours = sorted(
        neighbours,
        key=lambda x: tx_graph.degree(x),
        reverse=True
    )

    for neighbour in neighbours:

        if neighbour not in visited:

            visited.add(neighbour)
            queue.append(neighbour)

        if len(selected_transactions) + len(queue) >= N_TRANSACTIONS:
            break


# If the component was smaller than requested,
# supplement with other transaction nodes.
if len(selected_transactions) < N_TRANSACTIONS:

    for node in degree_sorted:

        if node not in selected_transactions:

            selected_transactions.append(node)

        if len(selected_transactions) >= N_TRANSACTIONS:
            break


selected_transactions = set(
    selected_transactions[:N_TRANSACTIONS]
)

print(
    f"Selected transactions: "
    f"{len(selected_transactions):,}"
)


# ============================================================
# BUILD VISUAL SUBGRAPH
# ============================================================

print("\nBuilding visual subgraph...")

selected_tx = selected_transactions


# ------------------------------------------------------------
# Wallet relationships belonging ONLY to selected transactions
# ------------------------------------------------------------

selected_participates = participates[
    participates["tail"].isin(selected_tx)
]

selected_outputs = has_output[
    has_output["head"].isin(selected_tx)
]


# ------------------------------------------------------------
# Transaction-to-transaction relationships
#
# IMPORTANT:
# Both ends must belong to the selected transaction set.
#
# This is the key fix compared with the previous script.
# ------------------------------------------------------------

selected_money_flow = money_flow[
    money_flow["head"].isin(selected_tx)
    &
    money_flow["tail"].isin(selected_tx)
]


print(
    f"Wallet participation triples: "
    f"{len(selected_participates):,}"
)

print(
    f"Wallet output triples: "
    f"{len(selected_outputs):,}"
)

print(
    f"Transaction flow triples: "
    f"{len(selected_money_flow):,}"
)


# ============================================================
# SELECT WALLETS
# ============================================================

wallet_candidates = set()

wallet_candidates.update(
    selected_participates["head"].dropna()
)

wallet_candidates.update(
    selected_outputs["tail"].dropna()
)

print(
    f"Connected wallets before limiting: "
    f"{len(wallet_candidates):,}"
)


# ------------------------------------------------------------
# Limit wallet count
#
# Prefer wallets that participate in several selected
# relationships, because they make the visual graph more
# informative.
# ------------------------------------------------------------

wallet_edges = pd.concat(
    [
        selected_participates[["head", "tail"]],
        selected_outputs[["head", "tail"]]
    ],
    ignore_index=True
)

wallet_degree = pd.concat(
    [
        wallet_edges["head"],
        wallet_edges["tail"]
    ]
).value_counts()


selected_wallets = set(
    wallet_degree.head(MAX_WALLETS).index
)

print(
    f"Wallets selected for visualisation: "
    f"{len(selected_wallets):,}"
)


# ============================================================
# FILTER FINAL EDGES
# ============================================================

final_participates = selected_participates[
    selected_participates["head"].isin(selected_wallets)
]

final_outputs = selected_outputs[
    selected_outputs["tail"].isin(selected_wallets)
]

final_money_flow = selected_money_flow


final_triples = pd.concat(
    [
        final_participates,
        final_outputs,
        final_money_flow
    ],
    ignore_index=True
)


print(
    f"\nFinal visual triples: "
    f"{len(final_triples):,}"
)

# ============================================================
# CALCULATE NODE POSITIONS IN PYTHON
# ============================================================

print("\nCalculating graph layout...")

layout_graph = nx.DiGraph()

# Add transaction nodes
for tx in selected_tx:
    layout_graph.add_node(
        str(tx),
        node_type="transaction"
    )

# Add wallet nodes
for wallet in selected_wallets:
    layout_graph.add_node(
        str(wallet),
        node_type="wallet"
    )

# Add the actual selected edges
for row in final_participates.itertuples(index=False):
    layout_graph.add_edge(
        str(row.head),
        str(row.tail)
    )

for row in final_outputs.itertuples(index=False):
    layout_graph.add_edge(
        str(row.head),
        str(row.tail)
    )

for row in final_money_flow.itertuples(index=False):
    layout_graph.add_edge(
        str(row.head),
        str(row.tail)
    )

# Calculate positions once in Python.
# PyVis will NOT have to calculate them in the browser.
positions = nx.spring_layout(
    layout_graph,
    seed=SEED,
    k=1.8,
    iterations=100,
    scale=800
)

print("Graph layout calculated.")

# ============================================================
# CREATE PYVIS NETWORK
# ============================================================

print("\nCreating interactive visualization...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# in_line prevents PyVis from depending on external JavaScript
# resources while the browser loads the graph.
net = Network(
    height="900px",
    width="100%",
    directed=True,
    bgcolor="#ffffff",
    font_color="#222222",
    cdn_resources="in_line"
    
)
net.toggle_physics(False)



# ============================================================
# ADD NODES
# ============================================================

# ------------------------------------------------------------
# Transaction nodes
# ------------------------------------------------------------

for tx in sorted(selected_tx):

    node_id = str(tx)

    x, y = positions[node_id]

    net.add_node(
        node_id,
        label="TX",
        title=f"Transaction\n{node_id}",
        shape="box",
        size=22,
        color="#ffb347",
        borderWidth=2,
        x=float(x),
        y=float(y),
        physics=False,
        fixed={
            "x": True,
            "y": True
        }
    )


# ------------------------------------------------------------
# Wallet nodes
# ------------------------------------------------------------

for wallet in sorted(selected_wallets):

    wallet_str = str(wallet)

    short_label = (
        wallet_str[:6]
        + "..."
        + wallet_str[-4:]
    )

    x, y = positions[wallet_str]

    net.add_node(
        wallet_str,
        label=short_label,
        title=f"Wallet\n{wallet_str}",
        shape="dot",
        size=14,
        color="#6fa8dc",
        borderWidth=1,
        x=float(x),
        y=float(y),
        physics=False,
        fixed={
            "x": True,
            "y": True
        }
    )

# ============================================================
# ADD EDGES
# ============================================================

for row in final_participates.itertuples(index=False):

    net.add_edge(
        str(row.head),
        str(row.tail),
        label="participates_in",
        title="participates_in",
        arrows="to",
    )


for row in final_outputs.itertuples(index=False):

    net.add_edge(
        str(row.head),
        str(row.tail),
        label="has_output",
        title="has_output",
        arrows="to",
    )


for row in final_money_flow.itertuples(index=False):

    net.add_edge(
        str(row.head),
        str(row.tail),
        label="money_flows_to",
        title="money_flows_to",
        arrows="to",
    )


# ============================================================
# NETWORK PHYSICS
# ============================================================
net.set_options(
    """
    {
      "nodes": {
        "font": {
          "size": 12
        }
      },

      "edges": {
        "font": {
          "size": 9,
          "align": "middle"
        },
        "smooth": false
      },

      "physics": {
        "enabled": false
      },

      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "dragNodes": false
      }
    }
    """
)


# ============================================================
# SAVE HTML AS UTF-8
# ============================================================

html = net.generate_html()
# Remove PyVis's loading overlay.
# We already calculated the node positions in Python,
# so the browser does not need a stabilization/loading screen.

html = html.replace(
    '<div id="loadingBar">',
    '<div id="loadingBar" style="display:none;">'
)

OUTPUT_HTML.write_text(
    html,
    encoding="utf-8"
)

OUTPUT_HTML.write_text(
    html,
    encoding="utf-8"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("KNOWLEDGE GRAPH VISUALIZER CREATED")
print("=" * 70)

print(
    f"Transaction nodes shown: "
    f"{len(selected_tx):,}"
)

print(
    f"Wallet nodes shown: "
    f"{len(selected_wallets):,}"
)

print(
    f"Total nodes shown: "
    f"{len(selected_tx) + len(selected_wallets):,}"
)

print(
    f"Total edges shown: "
    f"{len(final_triples):,}"
)

print("\nVisualization saved to:")
print(OUTPUT_HTML)

print("\nOpen in your browser:")
print("http://localhost:8000")

print("\nStart the local server with:")
print(
    f"python -m http.server 8000 "
    f"--directory {OUTPUT_DIR}"
)

print("\n" + "=" * 70)