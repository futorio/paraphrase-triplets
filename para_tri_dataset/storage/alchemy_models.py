from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String


Base = declarative_base()


class PhraseVectorJournal(Base):
    __tablename__ = "phrase_vector_journal"

    id = Column(Integer, primary_key=True)
    matrix_row_idx = Column(Integer, nullable=False)
    file_id = Column(Integer, nullable=False)


class PhraseMatrixFilenameJournal(Base):
    __tablename__ = "phrase_matrix_filename_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String)
