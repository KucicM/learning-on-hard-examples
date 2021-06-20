from typing import List
from data_engine.database.repository import LossRepository
from collections import deque


class HighLossIdProvider:
    _index_queue = deque()
    _database_host = ""
    _inference_mode = False
    __repository = None

    def __init__(self, database_host: str = None):
        self._database_host = self._database_host if self._database_host is not None else database_host

    def update_losses(self, losses: List[float]):
        self._repository.insert_or_replace(zip(self._queue(), losses))

    def add_index_to_queue(self, values):
        for value in values:
            self._index_queue.append(value)
            yield value

    def selective_backprop(self, cutoff) -> List[int]:
        return self._repository.selective_backprop(cutoff)

    def robust_selective_backprop(self, cutoff, std_num) -> List[int]:
        return self._repository.robust_selective_backporp(cutoff, std_num)

    def _queue(self):
        while len(self._index_queue) != 0:
            yield self._index_queue.popleft()

    @property
    def inference_mode(self) -> bool:
        return self._inference_mode

    @inference_mode.setter
    def inference_mode(self, value):
        self._inference_mode = value

    @property
    def _repository(self) -> LossRepository:
        if self.__repository is None:
            self.__repository = LossRepository(self._database_host)
        return self.__repository
