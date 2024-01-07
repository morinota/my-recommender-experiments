from pathlib import Path
from training_executor import TrainingExecutor
from configs.recbole_config import load_config_from_yamls
from atomic_coverter.atomic_file_fetcher import AtomicFileFetcher


def main() -> None:
    # TODO: feature storeから学習データをfetchし、atomic fileに変換する処理をtraining pipelineに含めたい
    dataset_name = "mind_training_small"
    raw_data_dir = Path("./feature_store/raw_data/")
    atomic_files_dir = Path("./feature_store/atomic_data/")
    atomic_file_fetcher = AtomicFileFetcher()
    atomic_file_fetcher.fetch(raw_data_dir, atomic_files_dir, dataset_name)

    # current_dirを出力
    print(f"[LOG] current_dir: {Path.cwd()}")

    model = "BPR"
    config_files_dir = Path("./training_pipelines/configs/BPR_configs/")

    config_dict = load_config_from_yamls(config_files_dir)

    executor = TrainingExecutor(model, dataset_name, config_dict)
    executor.run()


if __name__ == "__main__":
    main()
