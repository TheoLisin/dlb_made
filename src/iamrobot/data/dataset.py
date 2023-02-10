import os
import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, List, Dict


class CaptchaDataset(Dataset):
    """Captcha dataset."""

    def __init__(
        self,
        root_dir: Union[Path, str],
        names: Optional[List[str]] = None,
        encode_dct: Optional[Dict[str, str]] = None,
        transform=None,
        use_cache: bool = False,
    ):
        """Dataset class for captcha data.

        Args:
            root_dir (Union[Path, str]): Directory with all the images.
            names (Optional[List[str]], optional):
                full names (with format) of files to use.
                If None all images from root_dir will be used. Defaults to None.
            encode_dct (Dict[str, str], optional): dict to encode name for CTC loss. Defaults to None.
            transform (_type_, optional): Optional transform to be applied
                on a sample. Defaults to None.
            use_cache (bool, optional): preload all images to RAM.. Defaults to False.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.encode_dct = encode_dct

        if names is None:
            self.names = []
            for name in os.listdir(root_dir):
                rname = self.root_dir / name
                if not os.path.isdir(rname):
                    self.names.append(name)
        else:
            self.names = names

        if use_cache:
            self.data = self._load_data()
        else:
            self.data = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.data is not None:
            img = self.data[idx]["img"]
            name = self.data[idx]["name"]
            enc_name = self.data[idx]["enc_name"]
        else:
            name = self.names[idx]
            path = self.root_dir / name
            img = cv2.imread(str(path))
            name = name.split(".")[0]
            enc_name = self._encode(name.split(".")[0])

        if self.transform:
            img = self.transform(img)

        return img, enc_name, name

    def _load_data(self):
        data = []
        for name in self.names:
            path = self.root_dir / name
            sample = {
                "img": cv2.imread(str(path)),
                "name": name.split(".")[0],
                "enc_name": self._encode(name.split(".")[0]),
            }
            data.append(sample)

        return data

    def _encode(self, name: str):
        if self.encode_dct is None:
            return name
        return torch.tensor([self.encode_dct[ch] for ch in name], dtype=torch.long)
