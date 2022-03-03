from pathlib import Path

import yaml

MAX_CONFIG_DEPTH = 3
NESTED_CONFIGS_KEY = "__nested_configs__"


class ValidationConfigError(ValueError):
    pass


def unpack_config(config_name: str, base_path: Path, depth: int = 0) -> dict:
    """
    Распаковывает YAML файл конфигурации в словарь.

    по специальному ключу __nested_configs__ указывается маппинг "название поля : название конфигурации".
    При распаковке конфигурации по названию поля указывается данные из файла, который находится в папке с названием
    "название поля" и в файле "название конфигурации.yaml" или "название конфигурации.yml"
    """
    if depth > MAX_CONFIG_DEPTH:
        raise ValueError("config max depth")

    yaml_config_filepath = (base_path / config_name).with_suffix(".yaml")
    yml_config_filepath = (base_path / config_name).with_suffix(".yml")

    if yaml_config_filepath.exists() and yml_config_filepath.exists():
        raise ValueError(f"both {yaml_config_filepath} and {yml_config_filepath} exists")
    elif yaml_config_filepath.exists():
        config_filepath = yaml_config_filepath
    elif yml_config_filepath.exists():
        config_filepath = yml_config_filepath
    else:
        raise ValueError(f"config file {yaml_config_filepath} or {yml_config_filepath} not exists")

    stream = config_filepath.open(mode="r").read()
    raw_config_data = yaml.load(stream, Loader=yaml.CLoader)

    config_data = {k: v for k, v in raw_config_data.items() if k != "__nested_configs__"}

    nested_configs = raw_config_data.get("__nested_configs__", None)
    if nested_configs is None:
        return config_data

    for field_name, nested_config_name in nested_configs.items():
        nested_base_path = base_path / field_name
        nested_config_data = unpack_config(nested_config_name, nested_base_path, depth + 1)

        config_data[field_name] = nested_config_data

    return config_data
