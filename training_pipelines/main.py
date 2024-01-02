from pathlib import Path
from custom_models.nrms import NRMS
from recbole.model.general_recommender import BPR
from experiment_executor import ExperimentExecutor


def main() -> None:
    # TODO: feature storeから学習データをfetchし、atomic fileに変換する処理をtraining pipelineに含めたい
    atomic_files_dir = Path("training_pipelines/atomic_dataset/")
    # current_dirを出力
    print(f"[LOG] current_dir: {Path.cwd()}")

    executor = ExperimentExecutor()
    executor.run()


if __name__ == "__main__":
    model = BPR
    main(model)
