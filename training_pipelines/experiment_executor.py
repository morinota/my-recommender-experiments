from task_interface import TaskInterface
from logging import Logger, getLogger
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger


class ExperimentExecutor(TaskInterface):
    ATOMIC_FILE_DIR = Path("./training_pipelines/atomic_dataset/")

    def __init__(
        self,
        model: AbstractRecommender,
        dataset_name: str = "mind",
        config_dict: dict = {},
    ) -> None:
        self.model = model

        self.config = Config(
            model,
            dataset_name,
            config_dict=config_dict,
        )
        init_seed(self.config["seed"], self.config["reproducibility"])

        init_logger(self.config)
        self.logger = getLogger()
        self.logger.info(self.config)

    def run(self) -> None:
        # create dataset & filtering & preprocessing
        dataset = create_dataset(self.config)
        self.logger.info(dataset)

        # create dataloaders
        train_data, valid_data, test_data = data_preparation(self.config, dataset)

        # model loading and initialization
        model = self.model(self.config, dataset)
        self.logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(self.config, model)
        self.logger.info(trainer)

        # model training
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
        )

        # model evaluation
        test_result = trainer.evaluate(test_data)
        self.logger.info(f"Best valid result: {best_valid_result}")
        self.logger.info(f"Test result: {test_result}")

        # model parameter saving
        trainer.save()

    def _init_logger(self) -> Logger:
        pass
