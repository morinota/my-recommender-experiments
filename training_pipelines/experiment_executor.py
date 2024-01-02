from training_pipelines.task_interface import TaskInterface
from logging import getLogger
from pathlib import Path
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

from recbole.model.abstract_recommender import AbstractRecommender
from recbole.trainer import Trainer
from config.recbole_config import (
    EnvironmentSettings,
    DataSettings,
    TrainingSettings,
    EvaluationSettings,
)
from recbole.utils import init_seed


class ExperimentExecutor(TaskInterface):
    ATOMIC_FILE_DIR = Path("./training_pipelines/atomic_dataset/")

    def __init__(
        self,
        model: AbstractRecommender,
        dataset_name: str = "mind",
        model_hyper_params: dict = {},
    ) -> None:
        self.model = model

        recbole_config_dict = {
            **EnvironmentSettings(
                data_path=str(self.ATOMIC_FILE_DIR),
            ).to_dict(),
            **DataSettings().to_dict(),
            **TrainingSettings().to_dict(),
            **EvaluationSettings().to_dict(),
            **model_hyper_params,
        }
        self.config = Config(
            model,
            dataset_name,
            config_dict=recbole_config_dict,
        )
        init_seed(self.config["seed"], self.config["reproducibility"])

        self.logger = getLogger()

    def run(self) -> None:
        # create dataset & filtering & preprocessing
        dataset = create_dataset(self.config)

        # create dataloaders
        train_data, valid_data, test_data = data_preparation(self.config, dataset)

        # model loading and initialization
        model = self.model(self.config, dataset)

        # trainer loading and initialization
        trainer = Trainer(self.config, model)

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
