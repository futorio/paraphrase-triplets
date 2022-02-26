from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy import Column, Integer, String, create_engine


Base = declarative_base()


class Database:
    def __init__(self, engine: Engine, S):
        self._engine = engine
        self._Session = S

    @classmethod
    def from_url(cls, url: str, echo: bool = False) -> 'Database':
        engine = create_engine(url, echo=echo)
        Session = sessionmaker(engine, expire_on_commit=False)
        return cls(engine, Session)

    def create_all(self):
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session_scope(self) -> Session:
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()


class PhraseVectorJournal(Base):
    __tablename__ = 'phrase_vector_journal'

    id = Column(Integer, primary_key=True)
    matrix_row_idx = Column(Integer, nullable=False)
    file_id = Column(Integer, nullable=False)


class PhraseMatrixFilenameJournal(Base):
    __tablename__ = 'phrase_matrix_filename_journal'

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
