from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String

Base = declarative_base()


class ParaphraserPlusDataset(Base):
    __tablename__ = "paraphraser_plus_dataset"

    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    group_id = Column(Integer, nullable=False, index=True)
