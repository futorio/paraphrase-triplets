import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from para_tri_dataset.alchemy_utils import Database
from para_tri_dataset.config import Config

from para_tri_dataset.paraphrase_dataset import get_dataset_from_config
from para_tri_dataset.paraphrase_dataset.para_phraser_plus import ParaPhraserPlusSQLDataset
from para_tri_dataset.paraphrase_dataset.para_phraser_plus.base import ParaPhraserPlusPhrase

from para_tri_dataset.paraphrase_dataset.para_phraser_plus.alchemy_models import Base, ParaphraserPlusDataset


@pytest.fixture
def phrases():
    return [
        ParaPhraserPlusPhrase(id=0, text="foo"),
        ParaPhraserPlusPhrase(id=1, text="baz"),
        ParaPhraserPlusPhrase(id=2, text="bar"),
        ParaPhraserPlusPhrase(id=3, text="biz"),
    ]


@pytest.fixture
def paraphrases(phrases):
    return [
        (phrases[0], phrases[1]),
        (phrases[2], phrases[3]),
    ]


def create_database(db: Database, dataset_paraphrases):
    db.create_all(Base)

    with db.session_scope() as session:
        for group_id, phrases in enumerate(dataset_paraphrases):
            session.add_all([ParaphraserPlusDataset(id=p.id, text=p.text, group_id=group_id) for p in phrases])

        session.commit()


@pytest.fixture
def dataset_config(tmp_path, paraphrases):
    dataset_filepath = tmp_path / 'test-paraphraser-plus.sqlite'
    db_url = f"sqlite:///{dataset_filepath.absolute()}"

    engine = create_engine(db_url)
    Session = sessionmaker(engine, expire_on_commit=False)

    db = Database(engine, Session)
    create_database(db, paraphrases)

    cfg = Config(name="paraphrase_plus_sql", data={'db_url': db_url}, nested_configs=[])
    yield cfg

    dataset_filepath.unlink()


@pytest.fixture
def in_memory_database(paraphrases) -> Database:
    engine = create_engine("sqlite:///")
    Session = sessionmaker(engine, expire_on_commit=False)
    db = Database(engine, Session)

    create_database(db, paraphrases)
    return db


@pytest.fixture
def in_memory_dataset(in_memory_database) -> ParaPhraserPlusSQLDataset:
    return ParaPhraserPlusSQLDataset(in_memory_database)


def test_load_from_config(dataset_config):
    _ = ParaPhraserPlusSQLDataset.from_config(dataset_config)
    _ = get_dataset_from_config(dataset_config)

    with pytest.raises(ValueError):
        dataset_config.name = "foo"
        _ = get_dataset_from_config(dataset_config)


def test_get_phrase_by_id(in_memory_dataset, phrases):

    for phrase in phrases:
        dataset_phrase = in_memory_dataset.get_phrase_by_id(phrase.id)
        assert phrase == dataset_phrase


def test_iterate_phrases(in_memory_dataset, phrases):

    for orig_phrase, dataset_phrase in zip(phrases, in_memory_dataset.iterate_phrases()):
        assert orig_phrase == dataset_phrase

    for orig_phrase, dataset_phrase in zip(phrases[1:], in_memory_dataset.iterate_phrases(offset=1)):
        assert orig_phrase == dataset_phrase


def test_iterate_phrases_ids(in_memory_dataset, phrases):

    for orig_phrase, phrase_id in zip(phrases, in_memory_dataset.iterate_phrases_id()):
        assert orig_phrase.id == phrase_id

    for orig_phrase, phrase_id in zip(phrases[1:], in_memory_dataset.iterate_phrases_id(offset=1)):
        assert orig_phrase.id == phrase_id


def test_get_paraphrases(in_memory_dataset, paraphrases):

    for phrases in paraphrases:
        for i in range(len(phrases)):
            phrase, phrase_paraphrases = phrases[i], [p for j, p in enumerate(phrases) if j != i]
            dataset_paraphrases = in_memory_dataset.get_paraphrases(phrase)

            assert phrase_paraphrases == dataset_paraphrases


def test_get_paraphrases_id(in_memory_dataset, paraphrases):

    for phrases in paraphrases:
        for i in range(len(phrases)):
            phrase_id, phrase_paraphrases_id = phrases[i].id, [p.id for j, p in enumerate(phrases) if j != i]
            dataset_paraphrases_id = in_memory_dataset.get_paraphrases_id(phrase_id)

            assert phrase_paraphrases_id == dataset_paraphrases_id
