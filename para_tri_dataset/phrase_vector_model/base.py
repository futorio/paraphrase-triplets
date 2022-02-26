import abc
from dataclasses import dataclass
from typing import Sequence, Any, Dict

import numpy as np
import torch

from para_tri_dataset.paraphrase_dataset.base import Phrase


@dataclass
class AbstractDataclass(abc.ABC):
    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


@dataclass
class PhraseVector(AbstractDataclass):
    id: Any
    body: Any


@dataclass
class PhraseNumpyVector(PhraseVector):
    id: int
    body: np.array


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
    def to_device(self, device: torch.device):
        pass

    @abc.abstractmethod
    def create_phrases_vectors(self, phrases: Sequence[Phrase], device: torch.device) -> Sequence[PhraseVector]:
        pass
