"""
Датасет парафраз ParaPhraserPlus с сайта http://paraphraser.ru/
"""

import json
import os
import zipfile
from typing import Tuple, TypedDict, List, Dict, Generator, Sequence

from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset
from para_tri_dataset.config import Config
from para_tri_dataset.paraphrase_dataset.para_phraser_plus.base import ParaPhraserPlusPhrase


class SerializedRecordType(TypedDict):
    rubric: str
    date: str
    headlines: List[str]


SerializedDatasetType = Dict[str, SerializedRecordType]


def parse_json_dataset(
    dataset: SerializedDatasetType,
) -> Generator[Tuple[ParaPhraserPlusPhrase, Tuple[int, ...]], None, None]:

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
        for idx in range(offset, len(self.phrases)):
            yield idx

    def iterate_phrases(self, offset: int = 0) -> Generator[ParaPhraserPlusPhrase, None, None]:
        for phrase_idx in self.iterate_phrases_id(offset):
            yield self.phrases[phrase_idx]

    def iterate_paraphrases_id(self, offset: int = 0) -> Generator[Tuple[int, ...], None, None]:

        group_idx, visited_idx = 0, set()
        for r_idx, r in enumerate(self.phrases_relations):
            if r_idx in visited_idx:
                continue

            if offset > group_idx:
                continue

            yield r_idx, *r

            group_idx += 1
            visited_idx.update(set(r))

    def iterate_paraphrases(self, offset: int = 0) -> Generator[Tuple[ParaPhraserPlusPhrase, ...], None, None]:

        for phrases_idx in self.iterate_paraphrases_id(offset):
            phrases = tuple(self.get_phrase_by_id(p_id) for p_id in phrases_idx)

            yield phrases

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
            raise ValueError(f"not fount phrase by id {phrase.id}") from err

        return tuple(self.get_phrase_by_id(p_id) for p_id in paraphrases_ids)

    def get_paraphrases_id(self, phrase_id: int) -> Sequence[int]:
        """Возвращает id фраз, которые являются парафразами данного"""
        try:
            return self.phrases_relations[phrase_id]
        except IndexError as err:
            raise ValueError(f"not found phrase by id {phrase_id}") from err
