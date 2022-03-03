"""
Абстрактные классы для описания датасета
"""
import abc
from dataclasses import dataclass
from typing import Any, Generator, Dict, Sequence


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
    @staticmethod
    @abc.abstractmethod
    def get_name() -> str:
        pass

    @abc.abstractmethod
    def size(self) -> int:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ParaphraseDataset":
        pass

    @abc.abstractmethod
    def iterate_phrases(self, offset: int = 0) -> Generator[Phrase, None, None]:
        pass

    @abc.abstractmethod
    def get_phrase_by_id(self, phrase_id: Any) -> Phrase:
        pass

    @abc.abstractmethod
    def get_paraphrases(self, phrase) -> Sequence[Phrase]:
        pass
