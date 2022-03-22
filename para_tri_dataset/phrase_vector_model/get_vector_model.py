"""
Инициализация векторной модели
"""
from typing import Dict

from para_tri_dataset.config import Config
from para_tri_dataset.phrase_vector_model.sberbank_models import SbertLargeMTNLU
from para_tri_dataset.phrase_vector_model.base import PhraseVectorModel

VECTOR_MODELS = [SbertLargeMTNLU]

MODELS_NAMES_MAPPING: Dict[str, PhraseVectorModel] = {model_cls.get_name(): model_cls for model_cls in VECTOR_MODELS}


def get_vector_model_from_config(cfg: Config):
    try:
        model_cls = MODELS_NAMES_MAPPING[cfg.name]
    except KeyError:
        raise ValueError(f'unknown vector model name "{cfg.name}"')

    return model_cls.from_config(cfg)
