from pathlib import Path

import pandas as pd
import networkx as nx


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "AddrAddr_edgelist.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "bitcoin_graph.graphml"


def main():
    print("Loading Bitcoin address interactions...")

    # Load address-to-address edges
    edges = pd.read_csv(DATA_PATH)

    # Remove duplicate edges
    edges = edges.drop_duplicates()

    # Remove self-loops
    edges = edges[edges["input_address"] != edges["output_address"]]

    print(f"Cleaned edge list: {len(edges):,} edges")

    # Build a directed graph
    graph = nx.from_pandas_edgelist(
        edges,
        source="input_address",
        target="output_address",
        create_using=nx.DiGraph()
    )

    # Graph statistics
    print("\nKnowledge Graph Statistics")
    print("--------------------------")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Edges: {graph.number_of_edges():,}")
    print(f"Weakly connected components: {nx.number_weakly_connected_components(graph):,}")

    # Save graph
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, OUTPUT_PATH)

    print(f"\nGraph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()