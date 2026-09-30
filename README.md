# Bitcoin Transaction Knowledge Graph

This repository contains the code and results for the Knowledge Graphs portfolio project.

## Project structure

- `src/` contains the scripts used to prepare the graph, create the data split, perform logical reasoning, and run graph queries.
- `results/` contains the outputs and results used in the portfolio report.
- `requirements.txt` contains the Python dependencies.
- `data/` is not included in the submission because the original dataset is large and publicly available.

## 1. Install requirements

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## 2. Download the data

Download the Elliptic++ Transactions Dataset from the official dataset repository:

https://github.com/git-disl/EllipticPlusPlus

The dataset can be downloaded from the provided Google Drive link:

https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l?usp=sharing

For this project, the required file is:

`txs_edgelist.csv`

Place it at:

`data/raw/txs_edgelist.csv`

The raw dataset is intentionally not included in the repository because of its size.

## 3. Prepare the Knowledge Graph

Run:

```bash
python src/prepare_triples.py
```

This prepares the transaction-to-transaction Knowledge Graph and creates the processed triple file used for the experiments.

## 4. Create the train/validation/test split

Run:

```bash
python src/split_data.py
```

This creates the entity-covered training, validation, and test splits used for link prediction.

## 5. Run logical reasoning

Run:

```bash
python src/logical_reasoning.py
```

This applies the two-hop reasoning rule used in the project:

`A -> B` and `B -> C` implies `A -> C`

The resulting analysis is used in the logical reasoning section of the portfolio.

## 6. Run Knowledge Graph queries

Run:

```bash
python src/kg_queries.py
```

This runs the graph-query examples used in the project, including direct and two-hop transaction relationships.

The query output is saved under:

`results/kg_queries/`

## 7. TransE link prediction

The link-prediction experiment uses TransE through PyKEEN.

The final trained model and results are available in:

`results/transe_improved/`

The final configuration used in the portfolio was:

- Embedding dimension: 128
- Batch size: 1024
- Epochs: 30
- Learning rate: 0.001
- Negative samples: 10
- Random seed: 42
- Device: CPU

The reported evaluation results are also included in the portfolio report.

## 8. Results

The `results/` directory contains the outputs used for the final portfolio.

The main results include:

- TransE link-prediction results
- Logical reasoning results
- Knowledge Graph query results

The final portfolio report provides the detailed methodology, analysis, results, and limitations.

## Reproduction order

For a full reproduction, use the following order:

1. Install the requirements.
2. Download `txs_edgelist.csv`.
3. Place it in `data/raw/`.
4. Run `prepare_triples.py`.
5. Run `split_data.py`.
6. Run the TransE experiment.
7. Run `logical_reasoning.py`.
8. Run `kg_queries.py`.