from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine


class Database:
    def __init__(self, engine: Engine, S):
        self._engine = engine
        self._Session = S

    @classmethod
    def from_url(cls, url: str, echo: bool = False) -> "Database":
        engine = create_engine(url, echo=echo)
        Session = sessionmaker(engine, expire_on_commit=False)
        return cls(engine, Session)

    def create_all(self, base_model):
        base_model.metadata.create_all(self._engine)

    @contextmanager
    def session_scope(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
