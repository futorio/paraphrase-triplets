import logging
from pathlib import Path

import pytest

from para_tri_dataset.para_phraser_plus.file_dataset import ParaPhraserPlusFileDataset, ParaPhraserPlusPhrase


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
    return ParaPhraserPlusPhrase(0, (1,), "foo")


@pytest.fixture
def phrase_b():
    return ParaPhraserPlusPhrase(1, (0,), "baz")


@pytest.fixture
def dataset(phrase_a, phrase_b) -> ParaPhraserPlusFileDataset:
    return ParaPhraserPlusFileDataset(
        (
            phrase_a,
            phrase_b,
        )
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


def test_get_paraphrases(dataset: ParaPhraserPlusFileDataset, phrase_a, phrase_b):
    paraphrases_a = dataset.get_paraphrases(phrase_a)
    assert len(paraphrases_a) == 1
    assert paraphrases_a[0] == phrase_b

    paraphrases_b = dataset.get_paraphrases(phrase_b)
    assert len(paraphrases_b) == 1
    assert paraphrases_b[0] == phrase_a


def test_get_phrase_by_idx(dataset: ParaPhraserPlusFileDataset, phrase_a, phrase_b):
    assert phrase_a == dataset.get_phrase_by_id(0)
    assert phrase_b == dataset.get_phrase_by_id(1)

    with pytest.raises(ValueError):
        dataset.get_phrase_by_id(2)
