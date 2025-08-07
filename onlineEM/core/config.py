from ml_collections import ConfigDict
from typing import Callable, Any
from .base import em_config, EM
from ..data.loaders import DataLoaderOutput


class Config(ConfigDict):
    em_config: em_config
    schedule: Callable[..., Any]
    model: EM
    data: DataLoaderOutput
    validation_interval: int
    log_interval: int
    checkpoint_dir: str
    data_path: str
    data_clean_path: str
    polyak_update_frequency: int
