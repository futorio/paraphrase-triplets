"""
Датасет парафраз ParaPhraserPlus с сайта http://paraphraser.ru/
"""

import json
import os
import zipfile
from dataclasses import dataclass
from typing import Tuple, TypedDict, List, Dict, Generator


class SerializedRecordType(TypedDict):
    rubric: str
    date: str
    headlines: List[str]


SerializedDatasetType = Dict[str, SerializedRecordType]


@dataclass
class ParaPhraserPlusPhrase:
    id: int  # id не уникальный!
    record_id: int
    text: str


@dataclass
class ParaPhraserPlusRecord:
    id: int
    rubric: str
    date: str
    phrases: Tuple[ParaPhraserPlusPhrase, ...]

    @classmethod
    def from_dict(cls, id_: int, d: SerializedRecordType) -> "ParaPhraserPlusRecord":
        phrases = tuple(ParaPhraserPlusPhrase(phrase_id, id_, text) for phrase_id, text in enumerate(d["headlines"]))
        return cls(id_, d["rubric"], d["date"], phrases)


class ParaPhraserPlusFileDataset:
    def __init__(self, records: Tuple[ParaPhraserPlusRecord, ...]):
        self.records = records

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

        records = tuple(ParaPhraserPlusRecord.from_dict(int(k), v) for k, v in dataset.items())
        return cls(records)

    @classmethod
    def from_json(cls, filepath: str) -> "ParaPhraserPlusFileDataset":
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"dataset json {filepath} not exists")

        try:
            with open(filepath, "r") as f:
                dataset: SerializedDatasetType = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"file {filepath} is not a json")

        records = tuple(ParaPhraserPlusRecord.from_dict(int(k), v) for k, v in dataset.items())
        return cls(records)

    def iterate_phrases(self) -> Generator[ParaPhraserPlusPhrase, None, None]:
        for record in self.records:
            yield from record.phrases

    def get_paraphrases(self, phrase: ParaPhraserPlusPhrase) -> Tuple[ParaPhraserPlusPhrase, ...]:
        """Возвращает список фраз, которые являются парафразами данного"""
        record: ParaPhraserPlusRecord = self.records[phrase.record_id]
        return tuple(p for p in record.phrases if p.id != phrase.id)
