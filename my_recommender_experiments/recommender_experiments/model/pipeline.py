from pathlib import Path
from typing import Any


from recommender_experiments.model.task_interface import TaskInterface


class PipelineTask(TaskInterface):
    def run(self) -> None:
        pass
        # Load MIND dataset
        train_loader = DatasetLoaderTask()
        dataset = train_loader.run()

        # Preprocess

        # Train

        # Evaluate
