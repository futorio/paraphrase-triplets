"""
Скрип для перевода фраз в векторное представление
"""

# TODO: чекпоинтинг
import logging
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from tqdm import tqdm

from para_tri_dataset.paraphrase_dataset import get_dataset_from_config
from para_tri_dataset.phrase_vector_model import get_vector_model_from_config

logger = logging.getLogger(__name__)


def iterable_chunk(seq, size: int):
    chunk = []
    for item in seq:
        if size > len(chunk):
            chunk.append(item)
            continue

        yield chunk
        chunk = []

    if len(chunk) > 0:
        yield chunk


@hydra.main(config_path="conf", config_name="vectorize_phrases")
def main(cfg: DictConfig):
    verbose = cfg.get('verbose', 0)
    if verbose > 0:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    logger.debug('load dataset')
    dataset = get_dataset_from_config(cfg["dataset"])
    logger.debug('done. Dataset name: "%s"', dataset.get_name())

    logger.debug('load phrase vector model')
    phrase_model = get_vector_model_from_config(cfg["vector_model"])
    logger.debug('done. Vector model name: "%s"', phrase_model.get_name())

    logger.debug('start vectorize phrases')

    output_path = Path(cfg.get('output_path', '.'))
    phrase_matrix_filename = f'd:{dataset.get_name()}:m:{phrase_model.get_name()}.npz'

    phrase_matrix_filepath = output_path / phrase_matrix_filename
    if phrase_matrix_filepath.exists():
        raise ValueError(f'phrase matrix {phrase_matrix_filepath} already exists')

    chunks = iterable_chunk(dataset.iterate_phrases(), cfg['batch_size'])

    pbar = tqdm(total=dataset.size())
    phrase_matrix = phrase_model.create_phrases_vectors(next(chunks))
    pbar.update(len(phrase_matrix))

    for chunk in chunks:
        phrase_chunk_matrix = phrase_model.create_phrases_vectors(chunk)
        phrase_matrix = np.vstack((phrase_matrix, phrase_chunk_matrix))
        pbar.update(len(phrase_chunk_matrix))

    pbar.close()

    np.savez_compressed(str(phrase_matrix_filepath), phrase_matrix)
    logger.debug('done matrix saved to %s', phrase_matrix_filepath)


if __name__ == "__main__":
    main()
