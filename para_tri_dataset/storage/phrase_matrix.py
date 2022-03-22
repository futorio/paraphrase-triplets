"""
Хранение векторов фраз на диске пачками по отдельным файлам

phrase vectors
--------------
phrase_id
matrix_row_idx
filename

StorageMetadata
{
    "vector_model": {"name": "...", "vector_dim": "...", "max_seq_len": ...},
    "dataset": {"name": "..."}
}
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from para_tri_dataset.storage.alchemy_models import PhraseVectorJournal, PhraseMatrixFilenameJournal, Database
from para_tri_dataset.phrase_vector_model.base import PhraseNumpyVector


@dataclass
class PhrasesVectorsStorageMetadata:
    vector_model_name: str
    phrase_vector_dim: int

    dataset_name: str

    def create_phrase_matrix_name(self, total_vectors_count, matrix_size) -> str:
        start, end = total_vectors_count, total_vectors_count + matrix_size - 1

        tmpl = "({vector_model})({dim})({dataset_name}){start}...{end}"
        return tmpl.format(
            vector_model=self.vector_model_name,
            dim=self.phrase_vector_dim,
            dataset_name=self.dataset_name,
            start=start,
            end=end,
        )

    def to_dict(self):
        return {
            "vector_model": {"name": self.vector_model_name, "dim": self.phrase_vector_dim},
            "dataset": {"name": self.dataset_name},
        }

    @classmethod
    def from_dict(cls, d) -> "PhrasesVectorsStorageMetadata":
        vm_cfg = d["vector_model"]
        return cls(vm_cfg["name"], vm_cfg["dim"], d["dataset"]["name"])


class PhrasesVectorsDiskStorage:
    """
    Хранение векторов фраз на диске в виде .npz файлов.

    """

    def __init__(
        self,
        base_path: Path,
        metadata: PhrasesVectorsStorageMetadata,
        storage_database: Database,
        checkpoint_every: int,
    ):

        self.base_path = base_path
        self.metadata = metadata
        self.storage_database = storage_database
        self.checkpoint_every = checkpoint_every

        self.phrase_vectors_buffer = []

    def close(self):
        if len(self.phrase_vectors_buffer) > 0:
            self.dump_buffer()

    @staticmethod
    def _add_phrases_matrix_to_journal(session: Session, filename: str, phrase_vector_ids: Sequence[int]):
        filename_journal_record = PhraseMatrixFilenameJournal(filename=filename)
        session.add(filename_journal_record)
        session.commit()

        file_id = filename_journal_record.id
        phrase_vector_records = []
        for row_idx, phrase_vector_id in enumerate(phrase_vector_ids):
            journal_record = PhraseVectorJournal(id=phrase_vector_id, matrix_row_idx=row_idx, file_id=file_id)
            phrase_vector_records.append(journal_record)

        session.add_all(phrase_vector_records)
        session.commit()

    @staticmethod
    def _get_vector_count(session: Session) -> int:
        return session.query(PhraseVectorJournal.id).count()

    @staticmethod
    def _get_phrase_vector_data(session: Session, phrase_id: int) -> Optional[Tuple[str, int]]:
        result = (
            session.query(PhraseVectorJournal, PhraseMatrixFilenameJournal)
            .filter(PhraseVectorJournal.file_id == PhraseMatrixFilenameJournal.id)
            .where(PhraseVectorJournal.id == phrase_id)
            .one()
        )

        if len(result) == 0:
            return None

        phrase_vector_record, phrase_matrix_record = result
        return phrase_matrix_record.filename, phrase_vector_record.matrix_row_idx

    def dump_buffer(self):
        if len(self.phrase_vectors_buffer) == 0:
            raise RuntimeError("buffer is empty")

        with self.storage_database.session_scope() as session:
            total_vector_count = self._get_vector_count(session)

        matrix_name = self.metadata.create_phrase_matrix_name(total_vector_count, len(self.phrase_vectors_buffer))
        matrix_filename = f"{matrix_name}.npy"
        tmp_matrix_filepath = self.base_path / ("tmp" + matrix_filename)
        matrix_filepath = self.base_path / matrix_filename

        phrase_matrix = np.vstack(tuple(p.body for p in self.phrase_vectors_buffer))

        np.save(str(tmp_matrix_filepath), phrase_matrix)
        with self.storage_database.session_scope() as session:
            try:
                ids = [pv.id for pv in self.phrase_vectors_buffer]
                self._add_phrases_matrix_to_journal(session, matrix_filename, ids)
            except Exception as ex:
                tmp_matrix_filepath.unlink(missing_ok=True)
                raise ex

        tmp_matrix_filepath.rename(matrix_filepath)

    def get_vector_count(self) -> int:
        with self.storage_database.session_scope() as session:
            vector_count = self._get_vector_count(session)

        return vector_count

    def add_phrase_vector(self, phrase_vector: PhraseNumpyVector):
        self.phrase_vectors_buffer.append(phrase_vector)
        if len(self.phrase_vectors_buffer) == self.checkpoint_every:
            self.dump_buffer()
            self.phrase_vectors_buffer = []

    def add_phrase_vectors(self, phrase_vectors: Sequence[PhraseNumpyVector]):
        for pv in phrase_vectors:
            self.add_phrase_vector(pv)

    def get_phrase_vector(self, phrase_id: int) -> PhraseNumpyVector:
        for pv in self.phrase_vectors_buffer:
            if pv.id == phrase_id:
                return pv

        with self.storage_database.session_scope() as session:
            data = self._get_phrase_vector_data(session, phrase_id)

        if data is None:
            raise ValueError(f"phrase vector {phrase_id} not found")

        matrix_filename, matrix_row_idx = data

        matrix_filepath = self.base_path / matrix_filename

        phrase_matrix = np.load(str(matrix_filepath))
        phrase_vector = phrase_matrix[matrix_row_idx]
        return PhraseNumpyVector(phrase_id, phrase_vector)


def _compare_metadata(metadata: PhrasesVectorsStorageMetadata, metadata_filepath: Path):
    """Сопоставление файла метаданных на диске, с переданными метаданными"""
    with metadata_filepath.open(mode="r") as f:
        try:
            stored_metadata = PhrasesVectorsStorageMetadata.from_dict(json.load(f))
        except json.JSONDecodeError:
            raise ValueError(f"metadata {metadata_filepath} is not a json file")

    if metadata != stored_metadata:
        msg = (
            f"metadata on disk {metadata_filepath} ({stored_metadata}) and "
            f"current metadata ({metadata}) does not equal"
        )
        raise ValueError(msg)


def create_phrase_vector_storage(
    path: str, vector_model_name: str, phrase_vector_dim: int, dataset_name: str, checkpoint_every: int
) -> PhrasesVectorsDiskStorage:
    base_path = Path(path)
    base_vectors_storage_path = base_path / "vectors"

    metadata_filepath = base_path / "metadata.json"
    phrase_journal_filepath = base_path / "phrase_vectors_journal.sqlite3"
    db_url = f"sqlite:///{phrase_journal_filepath}"

    metadata = PhrasesVectorsStorageMetadata(vector_model_name, phrase_vector_dim, dataset_name)

    if metadata_filepath.exists() and phrase_journal_filepath.exists():
        # validate metadata
        _compare_metadata(metadata, metadata_filepath)
        storage_db = Database.from_url(db_url)
    elif not (metadata_filepath.exists() or phrase_journal_filepath.exists()):
        # init database, create metadata.json, create vectors dir

        base_vectors_storage_path.mkdir()

        storage_db = Database.from_url(db_url)
        storage_db.create_all()

        with metadata_filepath.open(mode="w") as f:
            json.dump(metadata.to_dict(), f)

    else:
        raise ValueError()

    return PhrasesVectorsDiskStorage(base_vectors_storage_path, metadata, storage_db, checkpoint_every)
