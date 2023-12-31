from logging import getLogger
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from models.nrms import NRMS
from recbole.model.general_recommender import BPR
from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer
from atomic_coverter.atomic_file_fetcher import AtomicFileFetcher
from config.recbole_config import (
    EnvironmentSettings,
    DataSettings,
    TrainingSettings,
    EvaluationSettings,
)
from recbole.utils import init_logger, init_seed


def main(model: AbstractRecommender, model_hyper_params: dict = {}) -> None:
    # feature storeから学習データをfetchし、atomic fileに変換する
    atomic_files_dir = Path("atomic_dataset/mind")
    fetcher = AtomicFileFetcher()
    fetcher.fetch(atomic_files_dir)

    recbole_config_dict = {
        **EnvironmentSettings().to_dict(),
        **DataSettings().to_dict(),
        **TrainingSettings().to_dict(),
        **EvaluationSettings().to_dict(),
        **model_hyper_params,
    }
    config = Config(model=model, dataset="mind", config_dict=recbole_config_dict)
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model(config, dataset)
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
    )

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info(f"Best valid result: {best_valid_result}")
    logger.info(f"Test result: {test_result}")

    # model parameter saving
    trainer.save()


if __name__ == "__main__":
    model = BPR
    main(model)
