"""
Инициализация датасета из конфига
"""
from typing import Dict

from para_tri_dataset.paraphrase_dataset.para_phraser_plus import ParaPhraserPlusFileDataset
from para_tri_dataset.paraphrase_dataset.base import ParaphraseDataset

DATASETS = [ParaPhraserPlusFileDataset]

DATASET_NAMES_MAPPING: Dict[str, ParaphraseDataset] = {dataset_cls.get_name(): dataset_cls for dataset_cls in DATASETS}


def get_dataset_from_config(cfg):
    try:
        dataset_name = cfg["name"]
    except KeyError:
        raise ValueError('config has not attribute "name"')

    try:
        dataset_cls = DATASET_NAMES_MAPPING[dataset_name]
    except KeyError:
        raise ValueError(f'unknown dataset name "{dataset_name}"')

    return dataset_cls.from_config(cfg)
