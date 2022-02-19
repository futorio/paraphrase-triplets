from pathlib import Path

import pytest

from para_tri_dataset.para_phraser_plus.file_dataset import (
    ParaPhraserPlusFileDataset,
    ParaPhraserPlusRecord,
    ParaPhraserPlusPhrase,
)


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
def all_phrases():
    return ParaPhraserPlusPhrase(0, 0, "foo"), ParaPhraserPlusPhrase(0, 1, "baz"), ParaPhraserPlusPhrase(1, 1, "bar")


@pytest.fixture
def paraphrase_a():
    return ParaPhraserPlusPhrase(0, 1, "baz")


@pytest.fixture
def paraphrase_b():
    return ParaPhraserPlusPhrase(1, 1, "bar")


@pytest.fixture
def dataset() -> ParaPhraserPlusFileDataset:
    records = (
        ParaPhraserPlusRecord(0, "", "", (ParaPhraserPlusPhrase(0, 0, "foo"),)),
        ParaPhraserPlusRecord(
            1,
            "",
            "",
            (
                ParaPhraserPlusPhrase(0, 1, "baz"),
                ParaPhraserPlusPhrase(1, 1, "bar"),
            ),
        ),
    )
    return ParaPhraserPlusFileDataset(records)


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


def test_iterate_phrases(dataset: ParaPhraserPlusFileDataset, all_phrases):
    phrases = tuple(dataset.iterate_phrases())
    assert all_phrases == phrases


def test_get_paraphrases(dataset: ParaPhraserPlusFileDataset, paraphrase_a, paraphrase_b):
    paraphrases = dataset.get_paraphrases(paraphrase_a)

    assert len(paraphrases) == 1
    assert paraphrases[0] == paraphrase_b
