from abc import ABC
from pytorch_lightning import Callback
from torch.utils.data import Sampler
from data_engine.database.provider import HighLossIdProvider
from enum import Enum


class SamplerMode(Enum):
    TRADITIONAL_LEARNING = 1
    SELECTIVE_BACKPROP = 2
    ROBUST_SELECTIVE_BACKPROP = 3


class HighLossSampler(Sampler, Callback, ABC, HighLossIdProvider):
    __datasource = None

    def __init__(self, database_host: str, allowed_stale, cutoff, robust_std):
        super().__init__(None)
        self._database_host = database_host

        self._allowed_stale = allowed_stale
        self._stale_count = allowed_stale
        self._cutoff = cutoff
        self._robust_cutoff_std = robust_std

        if allowed_stale is None:
            self._sampling_mode = SamplerMode.TRADITIONAL_LEARNING
        elif robust_std is None:
            self._sampling_mode = SamplerMode.SELECTIVE_BACKPROP
        else:
            self._sampling_mode = SamplerMode.ROBUST_SELECTIVE_BACKPROP

    def __iter__(self):
        if self._sampling_mode == SamplerMode.TRADITIONAL_LEARNING:
            return iter(range(len(self)))

        if self.inference_mode:
            return iter(self._inference_selection())

        if self._stale_count >= self._allowed_stale:
            indices = self.add_index_to_queue(range(len(self)))
            self._stale_count = 1
            return iter(indices)

        self._stale_count += 1
        return iter([])  # nothing to update, running in stale configuration outside of inference

    def _inference_selection(self):
        if self._sampling_mode == SamplerMode.SELECTIVE_BACKPROP:
            return self.selective_backprop(self._cutoff)
        elif self._sampling_mode == SamplerMode.ROBUST_SELECTIVE_BACKPROP:
            return self.robust_selective_backprop(self._cutoff, self._robust_cutoff_std)

    def __len__(self):
        return len(self.datasource)  # todo FIX visual bug

    @property
    def datasource(self):
        return self.__datasource

    @datasource.setter
    def datasource(self, datasource):
        self.__datasource = datasource
