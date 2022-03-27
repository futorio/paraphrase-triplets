"""
Инициализация векторной модели
"""

from para_tri_dataset.config import Config
from para_tri_dataset.phrase_vector_model.sberbank_models import SbertLargeMTNLU

MODELS_NAMES_MAPPING = {"sbert_large_mt_nlu_ru": SbertLargeMTNLU}


def get_vector_model_from_config(cfg: Config):
    try:
        model_cls = MODELS_NAMES_MAPPING[cfg.name]
    except KeyError as err:
        raise ValueError(f'unknown vector model name "{cfg.name}"') from err

    return model_cls.from_config(cfg)
