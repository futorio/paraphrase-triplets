"""
Датасет парафраз ParaPhraserPlus с сайта http://paraphraser.ru/
"""

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Tuple, TypedDict, List, Dict, Generator

from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset, Phrase


class SerializedRecordType(TypedDict):
    rubric: str
    date: str
    headlines: List[str]


SerializedDatasetType = Dict[str, SerializedRecordType]


@dataclass
class ParaPhraserPlusPhrase(Phrase):
    id: int
    text: str

    paraphrases_ids: Tuple[int, ...]


class ParaPhraserPlusFileDataset(ParaphraseDataset):
    def __init__(self, phrases: Tuple[ParaPhraserPlusPhrase, ...]):
        self.phrases = phrases

    @staticmethod
    def get_name() -> str:
        return "paraphrase_plus_file"

    @staticmethod
    def parse_json_dataset(dataset: SerializedDatasetType) -> Generator[ParaPhraserPlusPhrase, None, None]:
        offset = 0
        for serialized_record in dataset.values():
            phrases_count = len(serialized_record["headlines"])

            for i, text in enumerate(serialized_record["headlines"]):
                phrase_id = offset + i
                paraphrases_ids = tuple(offset + j for j in range(phrases_count) if j != i)

                yield ParaPhraserPlusPhrase(phrase_id, text, paraphrases_ids)

            offset += phrases_count

    @classmethod
    def from_config(cls, cfg):
        try:
            name = cfg["name"]
        except KeyError:
            raise ValueError('config has not "name" attribute')

        dataset_name = cls.get_name()
        if dataset_name != name:
            msg = f'config dataset name "{name}" does not compare with dataset class name "{dataset_name}"'
            raise ValueError(msg)

        zip_filepath, json_filepath = cfg.get("zip_filepath"), cfg.get("json_filepath")
        if zip_filepath is None and json_filepath is None:
            raise ValueError(f"config does not contain zip_filepath or json_filepath")
        elif zip_filepath is not None and json_filepath is not None:
            raise ValueError(f"config must contain zip_filepath or json_filepath not both")

        elif zip_filepath is not None:
            return cls.from_zip(zip_filepath)
        else:
            return cls.from_json(json_filepath)

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

        phrases = tuple(cls.parse_json_dataset(dataset))
        return cls(phrases)

    @classmethod
    def from_json(cls, filepath: str) -> "ParaPhraserPlusFileDataset":
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"dataset json {filepath} not exists")

        try:
            with open(filepath, "r") as f:
                dataset: SerializedDatasetType = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"file {filepath} is not a json")

        phrases = tuple(cls.parse_json_dataset(dataset))
        return cls(phrases)

    def iterate_phrases(self, offset: int = 0) -> Generator[ParaPhraserPlusPhrase, None, None]:
        yield from self.phrases[offset:]

    def get_phrase_by_id(self, phrase_id: int) -> ParaPhraserPlusPhrase:
        try:
            return self.phrases[phrase_id]
        except IndexError:
            raise ValueError(f"not found phrase by id {phrase_id}")

    def get_paraphrases(self, phrase: ParaPhraserPlusPhrase) -> Tuple[ParaPhraserPlusPhrase, ...]:
        """Возвращает список фраз, которые являются парафразами данного"""
        return tuple(self.phrases[p_id] for p_id in phrase.paraphrases_ids)
