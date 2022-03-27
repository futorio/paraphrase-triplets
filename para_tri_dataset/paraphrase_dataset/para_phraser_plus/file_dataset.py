"""
Датасет парафраз ParaPhraserPlus с сайта http://paraphraser.ru/
"""

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Tuple, TypedDict, List, Dict, Generator, Any, Sequence

from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset, Phrase
from para_tri_dataset.config import Config


class SerializedRecordType(TypedDict):
    rubric: str
    date: str
    headlines: List[str]


SerializedDatasetType = Dict[str, SerializedRecordType]


@dataclass
class ParaPhraserPlusPhrase(Phrase):
    id: int
    text: str


def parse_json_dataset(dataset: SerializedDatasetType)\
        -> Generator[Tuple[ParaPhraserPlusPhrase, Tuple[int, ...]], None, None]:

    offset = 0
    for serialized_record in dataset.values():
        phrases_count = len(serialized_record["headlines"])

        for i, text in enumerate(serialized_record["headlines"]):
            phrase_id = offset + i
            paraphrases_ids = tuple(offset + j for j in range(phrases_count) if j != i)

            yield ParaPhraserPlusPhrase(phrase_id, text), paraphrases_ids

        offset += phrases_count


class ParaPhraserPlusFileDataset(ParaphraseDataset):
    def __init__(self, phrases: Tuple[ParaPhraserPlusPhrase, ...], phrases_relations: Tuple[Tuple[int, ...], ...]):
        self.phrases = phrases
        self.phrases_relations = phrases_relations

    @classmethod
    def from_config(cls, cfg: Config) -> "ParaPhraserPlusFileDataset":
        zip_filepath, json_filepath = cfg.get("zip_filepath", None), cfg.get("json_filepath", None)

        if zip_filepath is not None:
            return cls.from_zip(zip_filepath)
        else:
            return cls.from_json(json_filepath)

    @classmethod
    def from_json(cls, filepath: str) -> "ParaPhraserPlusFileDataset":
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"dataset json {filepath} not exists")

        try:
            with open(filepath, "r") as f:
                dataset: SerializedDatasetType = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"file {filepath} is not a json")

        phrases, phrases_relations = tuple(zip(*parse_json_dataset(dataset)))
        return cls(phrases, phrases_relations)

    @classmethod
    def from_zip(cls, filepath: str) -> "ParaPhraserPlusFileDataset":
        """
        загрузка датасета из zip архива в формате с сайта:

        http://paraphraser.ru/download/get?file_id=7
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"zip file {filepath} not exists")

        if not zipfile.is_zipfile(filepath):
            raise ValueError(f"file {filepath} is not a zip")

        with zipfile.ZipFile(filepath, "r") as zf:
            with zf.open("ParaPhraserPlus/ParaPhraserPlus.json", "r") as f:
                dataset: SerializedDatasetType = json.load(f)

        phrases, phrases_relations = tuple(zip(*parse_json_dataset(dataset)))
        return cls(phrases, phrases_relations)

    def size(self) -> int:
        return len(self.phrases)

    def iterate_phrases_id(self, offset: int = 0) -> Generator[int, None, None]:
        yield from (p.id for p in self.phrases)

    def iterate_phrases(self, offset: int = 0) -> Generator[ParaPhraserPlusPhrase, None, None]:
        yield from self.phrases[offset:]

    def get_phrase_by_id(self, phrase_id: int) -> ParaPhraserPlusPhrase:
        try:
            return self.phrases[phrase_id]
        except IndexError as err:
            raise ValueError(f"not found phrase by id {phrase_id}") from err

    def get_paraphrases(self, phrase: ParaPhraserPlusPhrase) -> Tuple[ParaPhraserPlusPhrase, ...]:
        """Возвращает список фраз, которые являются парафразами данного"""
        try:
            paraphrases_ids = self.phrases_relations[phrase.id]
        except IndexError as err:
            raise ValueError(f'not fount phrase by id {phrase.id}') from err

        return tuple(self.get_phrase_by_id(p_id) for p_id in paraphrases_ids)

    def get_paraphrases_id(self, phrase_id: Any) -> Sequence[int]:
        """Возвращает id фраз, которые являются парафразами данного"""
        try:
            return self.phrases_relations[phrase_id]
        except IndexError as err:
            raise ValueError(f"not found phrase by id {phrase_id}") from err
