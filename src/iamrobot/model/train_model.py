import typing as tp
import numpy as np

from logging import getLogger
from torch import nn, no_grad, Tensor, device as Device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.functional import char_error_rate
from tqdm import tqdm

from iamrobot.model.captcha_model import CaptchaModel
from iamrobot.utils.visualization import show_text_example


logger = getLogger(__name__)


def train(
    model: CaptchaModel,
    optim: Optimizer,
    device: Device,
    trainloader: DataLoader,
    testloader: DataLoader,
    clip_grad: float = 0.1,
    epochs: int = 100,
    verbose_ep: int = 10,
) -> tp.Tuple[tp.List[float], tp.List[float]]:
    model.to(device)

    ctc_loss = nn.CTCLoss(zero_infinity=True)

    train_loss_history = []
    test_loss_history = []

    dataiter = iter(testloader)
    test_img, _, test_labels = next(dataiter)
    test_img = test_img.to(device)[:8]

    show_text_example(model, test_img, test_labels, logger)

    for ep in range(1, epochs + 1):
        total_batches = 0
        total_loss = 0
        cer = 0

        model.train()

        for batch, codes, labels in tqdm(trainloader, desc=f"Epoch #{ep}"):
            total_batches += 1
            img = batch.to(device)
            logprobs = model(img)

            input_lengths = tuple(50 for _ in range(batch.shape[0]))
            target_lengths = tuple(5 for _ in range(batch.shape[0]))

            loss = ctc_loss(
                logprobs.transpose(0, 1),
                codes,
                input_lengths,
                target_lengths,
            )

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optim.step()
            total_loss += loss.item()

            cer += char_error_rate(
                model.decode_out(logprobs),
                labels,
            )

        train_loss_history.append(total_loss / total_batches)
        logger.info(f"Train Epoch {ep} | Mean batch loss: {total_loss / total_batches} | Mean CER: {cer / total_batches}")

        model.eval()
        with no_grad():
            total_batches = 0
            total_loss = 0
            cer = 0

            for batch, codes, labels in tqdm(testloader, desc=f"Epoch #{ep}"):
                total_batches += 1
                img = batch.to(device)
                logprobs = model(img)

                input_lengths = tuple(50 for _ in range(batch.shape[0]))
                target_lengths = tuple(5 for _ in range(batch.shape[0]))

                loss = ctc_loss(
                    logprobs.transpose(0, 1),
                    codes,
                    input_lengths,
                    target_lengths,
                )

                total_loss += loss.item()

                cer += char_error_rate(
                    model.decode_out(logprobs),
                    labels,
                )
        
            test_loss_history.append(total_loss / total_batches)
            logger.info(f"Test Epoch {ep} | Mean batch loss: {total_loss / total_batches} | Mean CER: {cer / total_batches}")

        if ep % verbose_ep == 0:
            show_text_example(model, test_img.to(device), test_labels, logger)
        
    return train_loss_history, test_loss_history