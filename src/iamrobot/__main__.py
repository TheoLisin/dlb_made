import logging
import os
import sys
import torch
import numpy as np

from pathlib import Path
from typing import Dict, Tuple
from dataclasses import asdict

from iamrobot.train_params import read_params
from iamrobot.data.loader import create_train_test_loader
from iamrobot.data.transformers import simple_transform
from iamrobot.model.captcha_model import CaptchaModel
from iamrobot.model.train_model import train
from iamrobot.paths import DATA, PARAMS


logger = logging.getLogger()
str_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter(
    "[%(asctime)s\t%(levelname)s\t%(name)s]: \n%(message)s",
)
logger.setLevel(logging.INFO)
str_handler.setFormatter(fmt)
logger.addHandler(str_handler)


def create_alphabet(path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    labels = []
    for name in os.listdir(path):
        if not os.path.isdir(path / name):
            label = name.split(".")[0]
            labels.append(label)

    chars = list("".join(labels))
    chars = sorted(np.unique(chars))

    id_to_ch = {i + 1: ch for i, ch in enumerate(chars)}
    id_to_ch[0] = "-"
    ch_to_id = {ch: i for i, ch in id_to_ch.items()}

    return id_to_ch, ch_to_id


def main():
    train_params = read_params(PARAMS)
    id_to_ch, ch_to_id = create_alphabet(DATA)

    trainloader, testloader = create_train_test_loader(
        DATA,
        simple_transform(),
        test_size=train_params.test_size,
        train_bs=train_params.train_batch_size,
        test_bs=train_params.test_batch_size,
        use_cache=train_params.use_cache,
        encode_dct=ch_to_id,
    )

    if train_params.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Cuda is not available.")
    elif train_params.device != "cpu":
        raise ValueError(f"Unknown device: {train_params.device}")

    device = torch.device(train_params.device)

    optim_func = getattr(torch.optim, train_params.optim, None)

    if optim_func is None:
        raise ValueError(
            f"Can't find optim '{train_params.optim}' in torch.optim module."
        )

    model = CaptchaModel(decode_dct=id_to_ch, **asdict(train_params.modelparams))

    optim = optim_func(model.parameters(), lr=train_params.lr)
    train_loss, test_loss = train(
        model=model,
        optim=optim,
        device=device,
        trainloader=trainloader,
        testloader=testloader,
        epochs=train_params.epochs,
        clip_grad=train_params.clip_grad,
    )

    # save model to mlflow
    # save graphics/reconstruction to mlflow


if __name__ == "__main__":
    main()
