"""
Скрип для перевода фраз в векторное представление
"""

import logging
import argparse
from pathlib import Path

import torch
import yaml

import cerberus
import cerberus.errors

from tqdm import tqdm

from para_tri_dataset.config import unpack_config, ValidationConfigError
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


CONFIG_VALIDATION_SCHEMA = {
    "batch_size": {"type": "integer", "min": 1},
    "device": {"type": "string", "allowed": ["cpu", "cuda"]},
    "verbose": {"type": "integer", "allowed": [0, 1]},
    "dataset": {"type": "dict"},
    "vector_model": {"type": "dict"},
    "vector_storage": {"type": "dict"},
}


def main():
    arg_parser = argparse.ArgumentParser("Перевод датасета парафраз в векторное представление")

    arg_parser.add_argument("--config-path", type=Path, required=True)
    arg_parser.add_argument("--config-name", type=str, required=True)

    args = arg_parser.parse_args()

    cfg = unpack_config(args.config_name, args.config_path)
    yaml_stream_cfg = yaml.dump(cfg, Dumper=yaml.CDumper)

    validator = cerberus.Validator(CONFIG_VALIDATION_SCHEMA, require_all=True)

    validation_success = validator(cfg)
    if not validation_success:
        raise ValidationConfigError(f"main config validation fail: {validator.errors}")

    verbose = cfg["verbose"]
    if verbose == 0:
        logging.basicConfig(level=logging.INFO)
    elif verbose == 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        raise ValueError(f"unknown verbose level {verbose}")

    logger.info(f"start with config:\n\n{yaml_stream_cfg}\n")

    logger.info("Stage: load dataset")
    dataset = get_dataset_from_config(cfg["dataset"])
    logger.info('done. Dataset name: "%s"', dataset.get_name())

    device_name = cfg["device"]
    device = torch.device(device_name)

    logger.info("Stage: load phrase vector model")
    phrase_model = get_vector_model_from_config(cfg["vector_model"])
    logger.info('DONE. Vector model name: "%s"', phrase_model.get_name())

    logger.info(f"Stage: push vector model to device {device}")
    phrase_model.to_device(device)

    logger.info("Stage: load phrase vector storage")

    # TODO: валидация дожна быть рядом со структурами
    try:
        vector_storage_path = cfg["vector_storage"]["output_path"]
    except KeyError:
        raise ValueError("config does not have vector_storage -> output_path")

    try:
        checkpoint_every = cfg["vector_storage"]["checkpoint_every"]
    except KeyError:
        raise ValueError("config does not have vector_storage -> checkpoint_every")

    phrase_vector_storage = create_phrase_vector_storage(
        vector_storage_path,
        phrase_model.get_name(),
        phrase_model.get_vector_size(),
        dataset.get_name(),
        checkpoint_every,
    )

    total_phrase_vectors = phrase_vector_storage.get_vector_count()

    phrases_iterator = iterable_chunk(dataset.iterate_phrases(total_phrase_vectors), cfg["batch_size"])

    logger.info(f"DONE. Total vectors in storage {total_phrase_vectors}")

    logger.info("Stage: start vectorize phrases")
    pbar = tqdm(initial=total_phrase_vectors, total=dataset.size())
    for phrases_batch in phrases_iterator:
        phrases_vectors_batch = phrase_model.create_phrases_vectors(phrases_batch, device)
        phrase_vector_storage.add_phrase_vectors(phrases_vectors_batch)

        pbar.update(len(phrases_batch))

    pbar.close()
    logger.info("DONE. Exit.")


if __name__ == "__main__":
    main()
