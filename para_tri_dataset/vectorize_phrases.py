"""
Скрип для перевода фраз в векторное представление
"""

# TODO: реализовать скрипт с конфигами Hydra
# TODO: способ конфигурации датасета
# TODO: чекпоинтинг

import argparse


def main():
    arg_parser = argparse.ArgumentParser(description='получение векторов фраз')

    dataset_group = arg_parser.add_argument_group(title='dataset')
    dataset_group.add_argument('--dataset-type', choices=['ParaPhraserPlus'], required=True)

    dataset_path_group = dataset_group.add_mutually_exclusive_group(required=True)
    dataset_path_group.add_argument('--zip-filepath', type=str)
    dataset_path_group.add_argument('--json-filepath', type=str)

    phrase_vector_group = arg_parser.add_argument_group(title='phrase vector')
    phrase_vector_group.add_argument('--model-type', type=str, choices=['sbert_large_mt_nlu_ru'], required=True)
    phrase_vector_group.add_argument('--model-path', type=str)

    args = arg_parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
