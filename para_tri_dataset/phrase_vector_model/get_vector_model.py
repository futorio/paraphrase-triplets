"""
Инициализация векторной модели
"""
from typing import Dict

from para_tri_dataset.phrase_vector_model.sberbank_models import SbertLargeMTNLU
from para_tri_dataset.phrase_vector_model.base import PhraseVectorModel

VECTOR_MODELS = [SbertLargeMTNLU]

MODELS_NAMES_MAPPING: Dict[str, PhraseVectorModel] = {model_cls.get_name(): model_cls for model_cls in VECTOR_MODELS}


def get_vector_model_from_config(cfg):
    try:
        model_name = cfg["name"]
    except KeyError:
        raise ValueError(f'config has not attribute "name"')

    try:
        model_cls = MODELS_NAMES_MAPPING[model_name]
    except KeyError:
        raise ValueError(f'unknown vector model name "{model_name}"')

    return model_cls.from_config(cfg)
