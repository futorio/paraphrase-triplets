"""
Скрип для перевода фраз в векторное представление
"""

# TODO: чекпоинтинг
import hydra
from omegaconf import DictConfig

from para_tri_dataset.paraphrase_dataset import get_dataset_from_config
from para_tri_dataset.phrase_vector_model import get_vector_model_from_config


@hydra.main(config_path="conf", config_name="vectorize_phrases")
def main(cfg: DictConfig):
    dataset = get_dataset_from_config(cfg['dataset'])
    phrase_model = get_vector_model_from_config(cfg['vector_model'])

    print(dataset)
    print(phrase_model)


if __name__ == '__main__':
    main()
