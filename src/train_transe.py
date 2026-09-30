from pathlib import Path
import json
import time

import torch
from pykeen.pipeline import pipeline
from pykeen.evaluation import RankBasedEvaluator


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "train.tsv"
)

VALID_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "valid.tsv"
)

TEST_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "link_prediction"
    / "test.tsv"
)

# Keep the improved experiment separate from the baseline
RESULTS_DIR = BASE_DIR / "results" / "transe_improved"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EMBEDDING_DIM = 128
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.001

# Number of corrupted triples generated for each
# positive training triple
NUM_NEGATIVES = 10

SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("IMPROVED TRANSE KNOWLEDGE GRAPH EMBEDDING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print("\nModel configuration:")
    print("  Model:            TransE")
    print(f"  Embedding dim:    {EMBEDDING_DIM}")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Learning rate:    {LEARNING_RATE}")
    print(f"  Negative samples: {NUM_NEGATIVES}")
    print(f"  Random seed:      {SEED}")

    # --------------------------------------------------------
    # Input files
    # --------------------------------------------------------

    print("\nInput:")
    print(f"  Training:   {TRAIN_FILE}")
    print(f"  Validation: {VALID_FILE}")
    print(f"  Test:       {TEST_FILE}")

    # --------------------------------------------------------
    # Check files exist
    # --------------------------------------------------------

    for path in [TRAIN_FILE, VALID_FILE, TEST_FILE]:

        if not path.exists():
            raise FileNotFoundError(
                f"Required input file does not exist:\n{path}"
            )

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING IMPROVED TRAINING")
    print("=" * 70)

    start_time = time.time()

    result = pipeline(

        # ----------------------------------------------------
        # Data
        # ----------------------------------------------------

        training=str(TRAIN_FILE),
        validation=str(VALID_FILE),
        testing=str(TEST_FILE),

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model="TransE",

        model_kwargs={
            "embedding_dim": EMBEDDING_DIM,
        },

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer="Adam",

        optimizer_kwargs={
            "lr": LEARNING_RATE,
        },

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        training_loop="sLCWA",

        training_kwargs={
            "num_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },

        # ----------------------------------------------------
        # Negative sampling
        #
        # IMPORTANT:
        # These arguments belong directly to pipeline(),
        # not inside training_kwargs.
        # ----------------------------------------------------

        negative_sampler="basic",

        negative_sampler_kwargs={
            "num_negs_per_pos": NUM_NEGATIVES,
        },

        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        random_seed=SEED,

        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        device=device,

        # ----------------------------------------------------
        # No early stopping
        #
        # Early stopping caused extremely expensive validation
        # evaluation in the previous experiment.
        # ----------------------------------------------------

        stopper=None,

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluator=RankBasedEvaluator,

        evaluation_kwargs={
            "batch_size": 32,
        },

        use_testing_data=True,
    )

    elapsed_time = time.time() - start_time

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print("IMPROVED TRANSE TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal training + evaluation time: "
        f"{elapsed_time / 60:.2f} minutes"
    )

    # ========================================================
    # SAVE PYKEEN RESULTS
    # ========================================================

    print("\nSaving model and results...")

    result.save_to_directory(RESULTS_DIR)

    # ========================================================
    # SAVE EXPERIMENT SUMMARY
    # ========================================================

    summary = {
        "experiment": "improved_transe",

        "model": "TransE",

        "embedding_dimension": EMBEDDING_DIM,

        "batch_size": BATCH_SIZE,

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "negative_samples_per_positive": NUM_NEGATIVES,

        "random_seed": SEED,

        "device": device,

        "elapsed_minutes": round(
            elapsed_time / 60,
            2
        ),
    }

    summary_path = (
        RESULTS_DIR / "experiment_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print(f"\nResults saved to:")
    print(RESULTS_DIR)

    print(f"\nExperiment summary:")
    print(summary_path)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()