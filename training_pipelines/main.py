from pathlib import Path
from custom_models.nrms import NRMS
from recbole.model.general_recommender import BPR
from training_executor import TrainingExecutor
from configs.recbole_config import load_config_from_yamls


def main() -> None:
    # TODO: feature storeから学習データをfetchし、atomic fileに変換する処理をtraining pipelineに含めたい
    atomic_files_dir = Path("./feature_store/atomic_data/")
    # current_dirを出力
    print(f"[LOG] current_dir: {Path.cwd()}")

    model = "NRMS"
    dataset_name = "mind_training_small"
    config_files_dir = Path("./training_pipelines/configs/BPR_configs/")

    config_dict = load_config_from_yamls(config_files_dir)

    executor = TrainingExecutor(model, dataset_name, config_dict)
    executor.run()


if __name__ == "__main__":
    main()
