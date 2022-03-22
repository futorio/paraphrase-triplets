"""
Инициализация датасета из конфига
"""
from typing import Dict

from para_tri_dataset.paraphrase_dataset.para_phraser_plus import ParaPhraserPlusFileDataset
from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset

from para_tri_dataset.config import Config

DATASETS = [ParaPhraserPlusFileDataset]

DATASET_NAMES_MAPPING: Dict[str, ParaphraseDataset] = {dataset_cls.get_name(): dataset_cls for dataset_cls in DATASETS}


def get_dataset_from_config(cfg: Config):
    try:
        dataset_cls = DATASET_NAMES_MAPPING[cfg.name]
    except KeyError:
        raise ValueError(f'unknown dataset name "{cfg.name}"')

    return dataset_cls.from_config(cfg)
