"""
Скрип для перевода фраз в векторное представление
"""

# TODO: чекпоинтинг
import logging

import hydra
import torch
from omegaconf import DictConfig

from tqdm import tqdm

from para_tri_dataset.paraphrase_dataset import get_dataset_from_config
from para_tri_dataset.phrase_vector_model import get_vector_model_from_config
from para_tri_dataset.storage import create_phrase_vector_storage

logger = logging.getLogger(__name__)


def iterable_chunk(seq, size: int):
    chunk = []
    for item in seq:
        chunk.append(item)
        if len(chunk) == size:
            yield chunk
            chunk = []

    if len(chunk) > 0:
        yield chunk


@hydra.main(config_path="conf", config_name="vectorize_phrases")
def main(cfg: DictConfig):
    verbose = cfg.get("verbose", 0)
    if verbose > 0:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    logger.debug("load dataset")
    dataset = get_dataset_from_config(cfg["dataset"])
    logger.debug('done. Dataset name: "%s"', dataset.get_name())

    device_name = cfg.get('device', 'cpu')
    device = torch.device(device_name)

    logger.debug("load phrase vector model")
    phrase_model = get_vector_model_from_config(cfg["vector_model"])
    logger.debug('done. Vector model name: "%s"', phrase_model.get_name())

    logger.debug(f"push vector model to device {device}")
    phrase_model.to_device(device)

    logger.debug('load phrase vector storage')
    try:
        vector_storage_path = cfg['vector_storage']['output_path']
    except KeyError:
        raise ValueError('config does not have vector_storage -> output_path')

    try:
        checkpoint_every = cfg['vector_storage']['checkpoint_every']
    except KeyError:
        raise ValueError('config does not have vector_storage -> checkpoint_every')

    phrase_vector_storage = create_phrase_vector_storage(vector_storage_path, phrase_model.get_name(),
                                                         phrase_model.get_vector_size(), dataset.get_name(),
                                                         checkpoint_every)

    total_phrase_vectors = phrase_vector_storage.get_vector_count()

    phrases_iterator = iterable_chunk(dataset.iterate_phrases(total_phrase_vectors), cfg["batch_size"])

    logger.debug(f'done. Total vectors in storage {total_phrase_vectors}')

    logger.debug("start vectorize phrases")
    pbar = tqdm(initial=total_phrase_vectors, total=dataset.size())
    for phrases_batch in phrases_iterator:
        phrases_vectors_batch = phrase_model.create_phrases_vectors(phrases_batch, device)
        phrase_vector_storage.add_phrase_vectors(phrases_vectors_batch)

        pbar.update(len(phrases_batch))

    pbar.close()


if __name__ == "__main__":
    main()
