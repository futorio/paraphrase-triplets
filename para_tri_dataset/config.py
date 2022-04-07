import copy
from pathlib import Path
from typing import Optional, Any

import yaml
from cerberus import Validator

MAX_CONFIG_DEPTH = 1
SYSTEM_CONFIG_FIELDS = ["__nested_configs__", "__val_schema__"]


class Config:
    def __init__(self, name: str, data: dict, nested_configs, type_: str = "__main__"):
        self.name = name
        self.type_ = type_
        self.data = data
        self.nested_configs = nested_configs

    def __repr__(self):
        data_part = ", ".join(f"{k}={v}" for k, v in self.data.items())
        nested_part = None
        if self.nested_configs is not None:
            nested_part = ",".join(ncfg.type_ for ncfg in self.nested_configs)

        return f"Config(name={self.name}, type={self.type_}, data=({data_part}), nested_configs=({nested_part}))"

    def get(self, key: str, default_value: Optional[Any] = None):
        return self.data.get(key, default_value)

    def get_nested_config(self, key: str) -> "Config":
        for nested_config in self.nested_configs:
            if nested_config.type_ == key:
                return nested_config
        else:
            raise ValueError(f"config {key} not found")

    def to_dict(self) -> dict:
        data = copy.deepcopy(self.data)
        for config in self.nested_configs:
            data[config.type_] = config.to_dict()

        return data


def create_config(base_path: Path, config_name: str, depth: int = 0, type_: str = "__main__"):
    if depth > MAX_CONFIG_DEPTH:
        raise ValueError(f"max depth config {depth}")

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

    config_data = {k: v for k, v in raw_config_data.items() if k not in SYSTEM_CONFIG_FIELDS}

    if "__val_schema__" not in raw_config_data:
        raise ValueError(f'config file {config_filepath} does not contain validation schema "__val_schema__"')

    validator = Validator(raw_config_data["__val_schema__"])
    if not validator(config_data):
        raise ValueError(validator.errors)

    nested_configs = []
    if "__nested_configs__" in raw_config_data:
        for nested_field_name, nested_config_name in raw_config_data["__nested_configs__"].items():
            nested_base_path = base_path / nested_field_name
            nested_configs.append(
                create_config(nested_base_path, nested_config_name, depth + 1, type_=nested_field_name)
            )

    return Config(config_name, config_data, nested_configs, type_)
