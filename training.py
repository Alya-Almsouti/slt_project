
from pickletools import optimize
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_news, VidText_Dataset

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

def main(args):
    utils.set_seed(args.seed)
    if args.use_wandb:
        wandb.init(
            project="Uni-Sign",
            name=args.run_name if hasattr(args, 'run_name') else 'FullClips',
            config=vars(args)
        )
    train_data = VidText_Dataset(path=train_label_paths[args.dataset], 
                                  args=args)
    print(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn, 
                                 drop_last=True)
                                
    dev_data = VidText_Dataset(path=dev_label_paths[args.dataset], 
                                args=args)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn)
    print('ALL good untill now')
if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)