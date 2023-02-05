import logging
import os
import numpy as np
import torch
import time

from transformers.utils import ExplicitEnum


class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


class OptimizerNames(ExplicitEnum):
    """
    Stores the acceptable string identifiers for optimizers.
    """

    ADAM = "adam"
    ADAMW = "adamw"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAFACTOR = "adafactor"
    ADAMW_BNB = "adamw_bnb_8bit"
    SGD = "sgd"
    NOVOGRAD = "novograd"
    LAMB = "lamb"


def log_dist(message: str, rank: int = 0, level: int = logging.INFO) -> None:
    if rank == -1:
        ranks = [i for i in range(int(os.environ["WORLD_SIZE"]))]
    else:
        ranks = [rank]
    my_rank = int(os.environ.get("RANK", "0"))
    if my_rank in ranks:
        if level == logging.INFO:
            logging.info(f"[Rank {my_rank}] {message}")
        if level == logging.WARNING:
            logging.warning(f"[Rank {my_rank}] {message}")
        if level == logging.ERROR:
            logging.error(f"[Rank {my_rank}] {message}")
        if level == logging.DEBUG:
            logging.debug(f"[Rank {my_rank}] {message}")
