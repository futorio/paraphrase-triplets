"""
Инициализация датасета из конфига
"""
from para_tri_dataset.paraphrase_dataset.para_phraser_plus import ParaPhraserPlusFileDataset, ParaPhraserPlusSQLDataset

from para_tri_dataset.config import Config

DATASET_NAMES_MAPPING = {
    "paraphrase_plus_file": ParaPhraserPlusFileDataset,
    "paraphrase_plus_sql": ParaPhraserPlusSQLDataset,
}


def get_dataset_from_config(cfg: Config):
    try:
        dataset_cls = DATASET_NAMES_MAPPING[cfg.name]
    except KeyError:
        raise ValueError(f'unknown dataset name "{cfg.name}"')

    return dataset_cls.from_config(cfg)
