from pathlib import Path

import pytest

from para_tri_dataset.paraphrase_dataset.para_phraser_plus.file_dataset import (
    ParaPhraserPlusFileDataset,
)
from para_tri_dataset.paraphrase_dataset.para_phraser_plus.base import ParaPhraserPlusPhrase


@pytest.fixture
def datadir(request) -> Path:
    test_filepath = Path(request.fspath)
    return test_filepath.parent / "data"


@pytest.fixture
def zip_dataset_filepath(datadir) -> str:
    return str(datadir / "dummy_paraphraser_plus.zip")


@pytest.fixture
def json_dataset_filepath(datadir) -> str:
    return str(datadir / "DummyParaPhraserPlus.json")


@pytest.fixture
def phrase_a():
    return ParaPhraserPlusPhrase(0, "foo")


@pytest.fixture
def phrase_b():
    return ParaPhraserPlusPhrase(1, "baz")


@pytest.fixture
def dataset(phrase_a, phrase_b) -> ParaPhraserPlusFileDataset:
    return ParaPhraserPlusFileDataset(
        (
            phrase_a,
            phrase_b,
        ),
        (
            (1,),
            (0,),
        ),
    )


def test_dataset_zip_load(zip_dataset_filepath: str, json_dataset_filepath: str):
    with pytest.raises(FileNotFoundError):
        _ = ParaPhraserPlusFileDataset.from_zip("")

    with pytest.raises(ValueError):
        _ = ParaPhraserPlusFileDataset.from_zip(json_dataset_filepath)

    _ = ParaPhraserPlusFileDataset.from_zip(zip_dataset_filepath)


def test_dataset_json_load(zip_dataset_filepath: str, json_dataset_filepath: str):
    with pytest.raises(FileNotFoundError):
        _ = ParaPhraserPlusFileDataset.from_json("")

    with pytest.raises(ValueError):
        _ = ParaPhraserPlusFileDataset.from_json(zip_dataset_filepath)

    _ = ParaPhraserPlusFileDataset.from_json(json_dataset_filepath)


def test_iterate_phrases(dataset: ParaPhraserPlusFileDataset, phrase_a, phrase_b):
    phrases = tuple(dataset.iterate_phrases())
    assert phrases == (
        phrase_a,
        phrase_b,
    )

    assert next(dataset.iterate_phrases(offset=1)) == phrase_b

    phrases_id = tuple(dataset.iterate_phrases_id())
    assert phrases_id == (
        0,
        1,
    )


def test_iterate_paraphrases(dataset: ParaPhraserPlusFileDataset, phrase_a, phrase_b):
    paraphrases_groups = list(dataset.iterate_paraphrases())

    assert len(paraphrases_groups) == 1

    paraphrases = paraphrases_groups[0]
    assert phrase_a in paraphrases
    assert phrase_b in paraphrases


def test_iterate_paraphrases_id(dataset: ParaPhraserPlusFileDataset, phrase_a, phrase_b):
    paraphrases_ids_groups = list(dataset.iterate_paraphrases_id())

    assert len(paraphrases_ids_groups) == 1

    paraphrases_ids = paraphrases_ids_groups[0]
    assert phrase_a.id in paraphrases_ids
    assert phrase_b.id in paraphrases_ids


def test_get_paraphrases(
    dataset: ParaPhraserPlusFileDataset, phrase_a: ParaPhraserPlusPhrase, phrase_b: ParaPhraserPlusPhrase
):

    paraphrases_a = dataset.get_paraphrases(phrase_a)
    assert len(paraphrases_a) == 1
    assert paraphrases_a[0] == phrase_b

    paraphrases_b = dataset.get_paraphrases(phrase_b)
    assert len(paraphrases_b) == 1
    assert paraphrases_b[0] == phrase_a

    paraphrases_id_a = dataset.get_paraphrases_id(phrase_a.id)
    assert len(paraphrases_id_a) == 1
    assert paraphrases_id_a[0] == phrase_b.id

    paraphrases_id_b = dataset.get_paraphrases_id(phrase_b.id)
    assert len(paraphrases_id_b) == 1
    assert paraphrases_id_b[0] == phrase_a.id


def test_get_phrase_by_idx(
    dataset: ParaPhraserPlusFileDataset, phrase_a: ParaPhraserPlusPhrase, phrase_b: ParaPhraserPlusPhrase
):

    assert phrase_a == dataset.get_phrase_by_id(0)
    assert phrase_b == dataset.get_phrase_by_id(1)

    with pytest.raises(ValueError):
        dataset.get_phrase_by_id(2)
