
from pickletools import optimize
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_news

import wandb
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from transformers import get_scheduler
from SLRT_metrics import translation_performance
from config import *
from typing import Iterable, Optional




if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)