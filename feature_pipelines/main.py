from pathlib import Path

from dataset_preparer import DatasetPreparer


def main():
    dataset_type = "mind_training_small"
    destination_dir = Path(f"./feature_store/atomic_data/{dataset_type}/")

    preparer = DatasetPreparer()
    atomic_files = preparer.run(dataset_type, destination_dir, False)

    for filepath in atomic_files:
        print(f"[LOG] {filepath} is successfully created.")


if __name__ == "__main__":
    main()
