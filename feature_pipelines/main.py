from pathlib import Path

from dataset_preparer import DatasetPreparer


def main():
    dataset_name = "mind_training_small"
    destination_dir = Path(f"./feature_store/raw_data/{dataset_name}/")

    preparer = DatasetPreparer()
    preparer.run(dataset_name, destination_dir, False)


if __name__ == "__main__":
    main()
