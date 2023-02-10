import os
import torch
from pathlib import Path
from typing import Any, Union, Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from iamrobot.data.dataset import CaptchaDataset


def create_train_test_loader(
    path: Union[str, Path],
    transforms: Any,
    train_bs: int,
    test_bs: int,
    test_size: float,
    seed: int = 42,
    **dataset_params,
) -> Tuple[DataLoader, DataLoader]:
    """Create test and train dataloaders.

    Args:
        path (Union[str, Path]): path to data.
        transforms (Any): Transforms for datasets.
        train_bs (int): train batch size.
        test_bs (int): test batch size.
        test_size (float): size of test part.
        seed (int, optional): Defaults to 42.

    Returns:
        Tuple[DataLoader, DataLoader]: train and test DataLoaders.
    """
    g = torch.Generator()
    g.manual_seed(seed)

    all_names = os.listdir(path)

    train_names, test_names = train_test_split(
        all_names,
        test_size=test_size,
        random_state=seed,
    )

    train_dataset = CaptchaDataset(
        path,
        train_names,
        transform=transforms,
        **dataset_params,
    )

    test_dataset = CaptchaDataset(
        path,
        test_names,
        transform=transforms,
        **dataset_params,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        generator=g,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
    )

    return train_loader, test_loader
