"""
Датасет парафраз ParaPhraserPlus, который располагается в базе данных (например sqlite)
"""
from typing import Generator, Sequence

import sqlalchemy.exc
from sqlalchemy import select, and_
from sqlalchemy.orm import Query

from para_tri_dataset.config import Config
from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset

from para_tri_dataset.alchemy_utils import Database
from para_tri_dataset.paraphrase_dataset.para_phraser_plus.alchemy_models import ParaphraserPlusDataset
from para_tri_dataset.paraphrase_dataset.para_phraser_plus.base import ParaPhraserPlusPhrase


class ParaPhraserPlusSQLDataset(ParaphraseDataset):
    def __init__(self, storage_db: Database, scroll_size: int = 100):
        self.storage_db = storage_db
        self.scroll_size = scroll_size

    @classmethod
    def from_config(cls, cfg: Config) -> "ParaPhraserPlusSQLDataset":
        scroll_size = cfg.get("scroll_size", 100)
        database = Database.from_url(cfg.get("db_url"))
        return cls(database, scroll_size)

    def size(self) -> int:
        with self.storage_db.session_scope() as session:
            return session.query(ParaphraserPlusDataset).count()

    def _scroll_rows(self, session, offset: int, fields) -> Query:
        return session.query(*fields).order_by(ParaphraserPlusDataset.id).limit(self.scroll_size + 1).offset(offset)

    def get_phrase_by_id(self, phrase_id: int) -> ParaPhraserPlusPhrase:
        with self.storage_db.session_scope() as session:
            try:
                row = (
                    session.query(ParaphraserPlusDataset.id, ParaphraserPlusDataset.text).where(
                        ParaphraserPlusDataset.id == phrase_id
                    )
                ).one()
            except sqlalchemy.exc.NoResultFound as ex:
                raise ValueError(f"not found phrase by id {phrase_id}") from ex

            return ParaPhraserPlusPhrase(id=row.id, text=row.text)

    @staticmethod
    def _get_paraphrases_query(phrase_id: int, fields) -> Query:
        group_id_sub = (
            select(ParaphraserPlusDataset.group_id).where(ParaphraserPlusDataset.id == phrase_id)
        ).scalar_subquery()

        result_query = select(*fields).where(
            and_(ParaphraserPlusDataset.group_id == group_id_sub, ParaphraserPlusDataset.id != phrase_id)
        )

        return result_query

    def get_paraphrases(self, phrase: ParaPhraserPlusPhrase) -> Sequence[ParaPhraserPlusPhrase]:
        with self.storage_db.session_scope() as session:
            fields = [ParaphraserPlusDataset.id, ParaphraserPlusDataset.text]
            try:
                rows = session.execute(self._get_paraphrases_query(phrase.id, fields)).all()
            except sqlalchemy.exc.NoResultFound as ex:
                raise ValueError(f"not fount phrase by id {phrase.id}") from ex

            return [ParaPhraserPlusPhrase(id=r.id, text=r.text) for r in rows]

    def get_paraphrases_id(self, phrase_id: int) -> Sequence[int]:
        with self.storage_db.session_scope() as session:
            fields = [ParaphraserPlusDataset.id]
            try:
                rows = session.execute(self._get_paraphrases_query(phrase_id, fields)).all()
            except sqlalchemy.exc.NoResultFound as ex:
                raise ValueError(f"not found phrase by id {phrase_id}") from ex

            return [r.id for r in rows]

    def iterate_phrases_id(self, start_offset: int = 0) -> Generator[int, None, None]:
        offset = start_offset
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id]
                rows = self._scroll_rows(session, offset, fields).all()

                yield from (row.id for i, row in enumerate(rows, start=1) if i < self.scroll_size + 1)

                if self.scroll_size + 1 > len(rows):
                    break

                offset += self.scroll_size

    def iterate_phrases(self, start_offset: int = 0) -> Generator[ParaPhraserPlusPhrase, None, None]:
        offset = start_offset
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id, ParaphraserPlusDataset.text]
                rows = self._scroll_rows(session, offset, fields).all()

                for row in rows[: self.scroll_size]:
                    yield ParaPhraserPlusPhrase(id=row.id, text=row.text)

                if self.scroll_size + 1 > len(rows):
                    break

                offset += self.scroll_size
