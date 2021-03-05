import logging
from .cost_repository import CostRepository
LOGGER = logging.getLogger(__name__)


class IndicesProvider:
    def __init__(self):
        self._cost_provider = CostRepository()
        pass
