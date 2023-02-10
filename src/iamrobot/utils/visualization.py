import matplotlib.pyplot as plt
import numpy as np
import sys
from typing import List, Optional
from torch import Tensor
from logging import Logger

from iamrobot.model.captcha_model import CaptchaModel


def visualize(imgs: List[Tensor], labels: List[str]):

    h = len(imgs) // 2 + len(imgs) % 2
    w = 2
    spb = f"{2}{h}"
    fig = plt.figure(figsize=(h*2, w*2), dpi=100)

    for ind, sample in enumerate(zip(imgs, labels)):
        img, label = sample
        img = img / 2 + 0.5
        npimg = img.detach().numpy()
        pos = int(f"{spb}{ind + 1}")
        ax = fig.add_subplot(pos)
        ax.set_title(label)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()


def show_example(model: CaptchaModel, test_img: Tensor):
    pred_labels = model.decode_out(model(test_img))
    visualize(test_img.cpu(), pred_labels)


def show_text_example(model: CaptchaModel, test_img: Tensor, test_labels: List[str], logger: Optional[Logger] = None) -> None:
    pred_labels = model.decode_out(model(test_img))
    ans = []
    for pl, l in zip(pred_labels, test_labels):
        ans.append(f"Predicted: {pl}; True: {l}")
    res = "\n".join(ans)
    if logger is None:
        sys.stdout.write(res)
        sys.stdout.write("\n")
    else:
        logger.info(res)
    