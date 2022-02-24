import abc
from typing import Sequence, Any, Dict

import numpy as np

from para_tri_dataset.paraphrase_dataset.base import Phrase


class PhraseVectorModel(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PhraseVectorModel":
        pass

    @abc.abstractmethod
    def get_vector_size(self) -> int:
        pass

    @abc.abstractmethod
    def create_phrases_vectors(self, phrases: Sequence[Phrase]) -> np.array:
        pass
