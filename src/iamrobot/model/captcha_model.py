import numpy as np
import torch
from torch import nn
from typing import Dict, List


class CaptchaModel(nn.Module):
    def __init__(self, decode_dct: Dict[int, str], in_channels: int = 3, in_h: int = 50, in_w: int = 200):
        num_of_classes = len(decode_dct)
        self.decode_dct = decode_dct

        super().__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels = 32,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            ),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=2,
            )
        )
        out_conv = 64 * (in_h // 4)
        self.conv_to_lstm = nn.Sequential(
            nn.Linear(out_conv, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        self.first_lstm = nn.LSTM(64, 128, bidirectional=True, dropout=0.2)
        self.second_lstm = nn.LSTM(256, 64, bidirectional=True, dropout=0.2)
        self.classification = nn.Sequential(
            nn.Linear(128, num_of_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_part(x)
        x = torch.transpose(x, -1, 1)
        new_shape = (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        x = torch.reshape(x, new_shape)
        x = self.conv_to_lstm(x)
        x, _ = self.first_lstm(x)
        x, _ = self.second_lstm(x)

        return self.classification(x)
    
    def decode_out(self, logprob: torch.Tensor, return_raw: bool = False) -> List[str]:
        dvc = torch.device('cpu')
        amax = logprob.to(dvc).detach().argmax(dim=-1).numpy()
        decoded = np.vectorize(self.decode_dct.get)(amax)

        words = []
        for code in decoded:
            word = []
            for i, ch in enumerate(code):
                if (i == 0 or ch != code[i - 1]) and ch != '-':
                    word.append(ch)
                elif i == 0 and ch == '':
                    word.append('')
            words.append("".join(word))

        if return_raw:
            return ["".join(raw) for raw in decoded]
        return words
