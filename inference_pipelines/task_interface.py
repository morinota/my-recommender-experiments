import abc
from typing import Any


class TaskInterface(abc.ABC):
    @abc.abstractmethod
    def run(self) -> Any:
        raise NotImplementedError
