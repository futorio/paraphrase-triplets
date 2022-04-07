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

    def _scroll_rows(self, session, offset: int, fields, order_by: list) -> Query:
        return session.query(*fields).order_by(*order_by).limit(self.scroll_size + 1).offset(offset)

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

    def iterate_phrases_id(self, offset: int = 0) -> Generator[int, None, None]:
        current_offset = offset
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id]
                rows = self._scroll_rows(session, current_offset, fields, order_by=[ParaphraserPlusDataset.id]).all()

            yield from (row.id for i, row in enumerate(rows, start=1) if i < self.scroll_size + 1)

            if self.scroll_size + 1 > len(rows):
                break

            current_offset += self.scroll_size

    def iterate_phrases(self, offset: int = 0) -> Generator[ParaPhraserPlusPhrase, None, None]:
        current_offset = offset
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id, ParaphraserPlusDataset.text]
                rows = self._scroll_rows(session, current_offset, fields, order_by=[ParaphraserPlusDataset.id]).all()

            for row in rows[: self.scroll_size]:
                yield ParaPhraserPlusPhrase(id=row.id, text=row.text)

            if self.scroll_size + 1 > len(rows):
                break

            current_offset += self.scroll_size

    def iterate_paraphrases_id(self, offset: int = 0) -> Generator[Sequence[int], None, None]:
        phrase_offset, paraphrases_offset = 0, offset
        total_paraphrases_groups = 0

        paraphrases_id = []
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id, ParaphraserPlusDataset.group_id]
                rows = self._scroll_rows(
                    session,
                    phrase_offset,
                    fields,
                    order_by=[ParaphraserPlusDataset.id, ParaphraserPlusDataset.group_id],
                ).all()

            for row in rows[: self.scroll_size]:
                if len(paraphrases_id) == 0 or row.group_id == paraphrases_id[0].group_id:
                    paraphrases_id.append(row)
                    continue

                if total_paraphrases_groups >= paraphrases_offset:
                    yield [r.id for r in paraphrases_id]

                total_paraphrases_groups += 1
                paraphrases_id = []

            if self.scroll_size + 1 > len(rows):
                break

            phrase_offset += self.scroll_size

    def iterate_paraphrases(self, offset: int = 0) -> Generator[Sequence[ParaPhraserPlusPhrase], None, None]:
        phrase_offset, paraphrases_offset = 0, offset
        total_paraphrases_groups = 0

        paraphrases = []
        while True:
            with self.storage_db.session_scope() as session:
                fields = [ParaphraserPlusDataset.id, ParaphraserPlusDataset.text, ParaphraserPlusDataset.group_id]
                rows = self._scroll_rows(
                    session,
                    phrase_offset,
                    fields,
                    order_by=[ParaphraserPlusDataset.id, ParaphraserPlusDataset.group_id],
                ).all()

            for row in rows[: self.scroll_size]:
                if len(paraphrases) == 0 or row.group_id == paraphrases[0].group_id:
                    paraphrases.append(row)
                    continue

                if total_paraphrases_groups >= paraphrases_offset:
                    yield [ParaPhraserPlusPhrase(r.id, r.text) for r in paraphrases]

                total_paraphrases_groups += 1
                paraphrases = []

            if self.scroll_size + 1 > len(rows):
                break

            phrase_offset += self.scroll_size
