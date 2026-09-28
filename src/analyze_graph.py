from pathlib import Path
import csv
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "AddrAddr_edgelist.csv"


class UnionFind:
    """Efficiently tracks connected components."""

    def __init__(self):
        self.parent = []
        self.size = []
        self.address_to_id = {}

    def get_id(self, address):
        if address not in self.address_to_id:
            node_id = len(self.parent)
            self.address_to_id[address] = node_id
            self.parent.append(node_id)
            self.size.append(1)
        return self.address_to_id[address]

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return

        # Attach smaller component to larger component
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]


def percentile(values, percentage):
    """Calculate a simple percentile without requiring NumPy."""
    if not values:
        return 0

    values = sorted(values)
    index = int((percentage / 100) * (len(values) - 1))
    return values[index]


def main():
    print("=" * 60)
    print("ELLIPTIC++ AddrAddr GRAPH ANALYSIS")
    print("=" * 60)

    if not DATA_FILE.exists():
        print(f"\nERROR: File not found:\n{DATA_FILE}")
        return

    print(f"\nDataset: {DATA_FILE}")
    print("\nReading graph...")
    print("This may take a little while.\n")

    union_find = UnionFind()

    in_degree = Counter()
    out_degree = Counter()

    edge_count = 0
    self_loops = 0

    with DATA_FILE.open(
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            source = row["input_address"]
            target = row["output_address"]

            edge_count += 1

            # Assign integer IDs to wallet addresses
            source_id = union_find.get_id(source)
            target_id = union_find.get_id(target)

            # Degree information
            out_degree[source] += 1
            in_degree[target] += 1

            # Self-loop
            if source == target:
                self_loops += 1

            # Weak connectivity
            union_find.union(source_id, target_id)

            if edge_count % 500_000 == 0:
                print(f"Processed {edge_count:,} edges...")

    # ----------------------------------------------------------
    # Connected components
    # ----------------------------------------------------------

    component_sizes = Counter()

    for node_id in range(len(union_find.parent)):
        root = union_find.find(node_id)
        component_sizes[root] += 1

    component_size_values = list(component_sizes.values())
    component_size_values.sort(reverse=True)

    # ----------------------------------------------------------
    # Degree statistics
    # ----------------------------------------------------------

    all_addresses = set(union_find.address_to_id.keys())

    total_addresses = len(all_addresses)

    total_degree = {
        address: in_degree[address] + out_degree[address]
        for address in all_addresses
    }

    degree_values = list(total_degree.values())

    # ----------------------------------------------------------
    # Results
    # ----------------------------------------------------------

    print("\n" + "=" * 60)
    print("GRAPH RESULTS")
    print("=" * 60)

    print(f"\nTotal edges:              {edge_count:,}")
    print(f"Unique wallet addresses: {total_addresses:,}")
    print(f"Self-loops:              {self_loops:,}")

    print("\nConnected components:")
    print(f"  Number of components:  {len(component_sizes):,}")
    print(f"  Largest component:     {component_size_values[0]:,} wallets")

    print("\nWallet degree statistics:")
    print(f"  Minimum degree:        {min(degree_values):,}")
    print(f"  Maximum degree:        {max(degree_values):,}")
    print(f"  Mean degree:           {sum(degree_values) / len(degree_values):.2f}")

    print("\nDegree percentiles:")
    print(f"  25th percentile:       {percentile(degree_values, 25):,}")
    print(f"  Median:                {percentile(degree_values, 50):,}")
    print(f"  75th percentile:       {percentile(degree_values, 75):,}")
    print(f"  90th percentile:       {percentile(degree_values, 90):,}")
    print(f"  95th percentile:       {percentile(degree_values, 95):,}")
    print(f"  99th percentile:       {percentile(degree_values, 99):,}")

    # ----------------------------------------------------------
    # Degree categories
    # ----------------------------------------------------------

    isolated = sum(1 for d in degree_values if d == 0)
    degree_1 = sum(1 for d in degree_values if d == 1)
    degree_2_5 = sum(1 for d in degree_values if 2 <= d <= 5)
    degree_6_10 = sum(1 for d in degree_values if 6 <= d <= 10)
    degree_11_plus = sum(1 for d in degree_values if d >= 11)

    print("\nDegree categories:")
    print(f"  Degree 0:              {isolated:,}")
    print(f"  Degree 1:              {degree_1:,}")
    print(f"  Degree 2–5:            {degree_2_5:,}")
    print(f"  Degree 6–10:           {degree_6_10:,}")
    print(f"  Degree 11+:            {degree_11_plus:,}")

    print("\nLargest connected components:")
    for rank, size in enumerate(component_size_values[:10], start=1):
        print(f"  {rank:2d}. {size:,} wallets")

    print("\n" + "=" * 60)
    print("GRAPH ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()