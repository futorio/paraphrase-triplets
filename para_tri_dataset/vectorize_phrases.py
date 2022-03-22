"""
Скрип для перевода фраз в векторное представление
"""

import logging
import argparse
import pprint
from pathlib import Path

import torch

from tqdm import tqdm

from para_tri_dataset.config import create_config
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


def main():
    arg_parser = argparse.ArgumentParser("Перевод датасета парафраз в векторное представление")

    arg_parser.add_argument("--config-path", type=Path, required=True)
    arg_parser.add_argument("--config-name", type=str, required=True)

    args = arg_parser.parse_args()

    cfg = create_config(args.config_path, args.config_name)

    verbose = cfg.get("verbose")
    if verbose == 0:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError(f"unknown verbose level {verbose}")

    pp_config = pprint.pformat(cfg.to_dict())
    logger.info(f"start with config:\n\n{pp_config}\n")

    logger.info("Stage: load dataset")
    dataset = get_dataset_from_config(cfg.get_nested_config("dataset"))

    logger.info('done. Dataset name: "%s"', dataset.get_name())

    device_name = cfg.get("device")
    device = torch.device(device_name)

    logger.info("Stage: load phrase vector model")
    phrase_model = get_vector_model_from_config(cfg.get_nested_config("vector_model"))
    logger.info('DONE. Vector model name: "%s"', phrase_model.get_name())

    logger.info(f"Stage: push vector model to device {device}")
    phrase_model.to_device(device)

    logger.info("Stage: load phrase vector storage")

    vector_storage_cfg = cfg.get_nested_config("vector_storage")
    if vector_storage_cfg.name != 'file_vector_storage':
        raise ValueError('only "file_vector_storage" support')

    phrase_vector_storage = create_phrase_vector_storage(
        vector_storage_cfg.get("output_path"),
        phrase_model.get_name(),
        phrase_model.get_vector_size(),
        dataset.get_name(),
        vector_storage_cfg.get("checkpoint_every"),
    )

    total_phrase_vectors = phrase_vector_storage.get_vector_count()

    phrases_iterator = iterable_chunk(dataset.iterate_phrases(total_phrase_vectors), cfg.get("batch_size"))

    logger.info(f"DONE. Total vectors in storage {total_phrase_vectors}")

    logger.info("Stage: start vectorize phrases")
    pbar = tqdm(initial=total_phrase_vectors, total=dataset.size())
    for phrases_batch in phrases_iterator:
        phrases_vectors_batch = phrase_model.create_phrases_vectors(phrases_batch, device)
        phrase_vector_storage.add_phrase_vectors(phrases_vectors_batch)

        pbar.update(len(phrases_batch))

    pbar.close()
    phrase_vector_storage.close()
    logger.info("DONE. Exit.")


if __name__ == "__main__":
    main()
