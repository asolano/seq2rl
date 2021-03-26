# coding: utf-8

import logging
import torch
from typing import (Optional,)


def create_logger(name: str, file_name: Optional[str] = None):
    logger = logging.getLogger(name)
    log_level = logging.DEBUG
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if file_name is not None:
        file_handler = logging.FileHandler(file_name)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info('Logger initialized.')

    return logger


def set_random_seed(seed: int, logger: logging.Logger) -> torch.Generator:
    generator = torch.manual_seed(seed)
    logger.debug(f'RNG seed set to {seed}')
    return generator


def get_device(cuda: bool, logger=None) -> torch.device:
    cuda_available = torch.cuda.is_available()
    # TODO device_count
    if cuda_available:
        if not cuda:
            if logger is not None:
                logger.info(f'CUDA device is available but not selected! Adding the CLI option is recommended.')
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if logger is not None:
        logger.debug(f'PyTorch device set to "{device}"')
    return device


def to_column_batches(data: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Convert sequential input data to `batch_size` columns, discarding excess elements.
    """
    n_batches = data.size(0) // batch_size
    trimmed = data.narrow(0, 0, n_batches * batch_size)
    batches = trimmed.view(batch_size, -1, data.size(1)).transpose(0, 1).contiguous()
    return batches.to(device)
