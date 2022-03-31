"""
Абстрактные классы для описания датасета
"""
import abc
from dataclasses import dataclass
from typing import Any, Generator, Sequence

from para_tri_dataset.config import Config


@dataclass
class AbstractDataclass(abc.ABC):
    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


# TODO: добавить в датасет возможность оперировать отношениями типа PhraseRelation(id, paraphrases)


@dataclass
class Phrase(AbstractDataclass):
    id: Any
    text: str


class ParaphraseDataset(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Config) -> "ParaphraseDataset":
        pass

    @abc.abstractmethod
    def size(self) -> int:
        pass

    @abc.abstractmethod
    def iterate_phrases(self, offset: int = 0) -> Generator[Phrase, None, None]:
        pass

    @abc.abstractmethod
    def iterate_phrases_id(self, offset: int = 0) -> Generator[Any, None, None]:
        pass

    @abc.abstractmethod
    def iterate_paraphrases_id(self, offset: int = 0) -> Generator[Sequence[Any], None, None]:
        pass

    @abc.abstractmethod
    def iterate_paraphrases(self, offset: int = 0) -> Generator[Sequence[Phrase], None, None]:
        pass

    @abc.abstractmethod
    def get_phrase_by_id(self, phrase_id: Any) -> Phrase:
        pass

    @abc.abstractmethod
    def get_paraphrases(self, phrase_id: Any) -> Sequence[Phrase]:
        pass

    @abc.abstractmethod
    def get_paraphrases_id(self, phrase_id: Any) -> Sequence[Any]:
        pass
