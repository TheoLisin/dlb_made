import yaml
from marshmallow_dataclass import class_schema
from dataclasses import dataclass
from typing import Union
from pathlib import Path


@dataclass
class ModelParams:
    in_channels: int
    in_h: int
    in_w: int


@dataclass
class TrainigParams:
    modelname: str
    modelparams: ModelParams
    optim: str
    lr: float
    device: str
    epochs: int
    test_batch_size: int
    train_batch_size: int
    use_cache: bool
    test_size: float
    clip_grad: float


TrainingPipelineSchema = class_schema(TrainigParams)


def read_params(path: Union[str, Path]) -> TrainigParams:
    """Read TrainigParams from yaml file."""
    with open(path, "r") as param_file:
        schema = TrainingPipelineSchema()
        tparams = schema.load(yaml.safe_load(param_file))

    return tparams
