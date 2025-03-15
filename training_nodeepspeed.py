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

def main(args):
    print(args)
    utils.set_seed(args.seed)

    # wandb.init(
    #     project="Uni-Sign",
    #     name=args.run_name if hasattr(args, 'run_name') else 'FullClips',
    #     config=vars(args)
    # )
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
    print("Creating dataset:")

    train_data = S2T_Dataset_news(path=train_label_paths[args.dataset], 
                                  args=args, phase='train')
    print(train_data)
    # Using default shuffling instead of DistributedSampler
    train_dataloader = DataLoader(train_data,
                                  batch_size=1, 
                                  num_workers=2, 
                                  collate_fn=train_data.collate_fn,
                                  shuffle=True,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")

    dev_data = S2T_Dataset_news(path=dev_label_paths[args.dataset], 
                                args=args, phase='dev')
    print(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=1,
                                num_workers=2, 
                                collate_fn=dev_data.collate_fn,
                                shuffle=False,
                                pin_memory=args.pin_mem)

    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
    print("Creating model:")
    model = Uni_Sign(args=args)
    print('hereno?')
    model.cuda()
    model.train()
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9} GB")
    # Optionally, watch model with wandb:
    # wandb.watch(model, log="all", log_freq=100)

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']
        ret = model.load_state_dict(state_dict, strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    
    # With distributed training removed, model_without_ddp is the same as model.
    model_without_ddp = model

    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model_without_ddp)
    
    if args.quick_break <= 0:
        args.quick_break = len(train_dataloader)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_epochs * len(train_dataloader) / args.gradient_accumulation_steps),
        num_training_steps=int(args.epochs * len(train_dataloader) / args.gradient_accumulation_steps),
    )
    
    output_dir = Path(args.output_dir)
    start_time = time.time()
    max_accuracy = 0

    if args.eval:
        print("📄 test result")
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        return 

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch, model_without_ddp=model_without_ddp)

        if args.output_dir:
            checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
            utils.save_on_master({
                'model': get_requires_grad_dict(model_without_ddp),
            }, checkpoint_path)
            
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")

        # wandb.log({
        #     "epoch": epoch,
        #     "BLEU-4": test_stats["bleu4"],
        #     "ROUGE": test_stats.get("rouge", 0),
        #     "Validation Loss": test_stats['loss']
        # })

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]
            if args.output_dir:
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)
            
        print(f'Max BLEU-4: {max_accuracy:.2f}%')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    # wandb.finish()

def train_one_epoch(args, model, data_loader, optimizer, epoch, model_without_ddp):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % 10 == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
        if (step + 1) % args.quick_break == 0:
            if args.output_dir:
                checkpoint_path = Path(args.output_dir) / 'checkpoint.pth'
                utils.save_on_master({
                    'model': get_requires_grad_dict(model_without_ddp),
                }, checkpoint_path)

        if target_dtype is not None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        stack_out = model(src_input, tgt_input)
        total_loss = stack_out['loss']
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    target_dtype = torch.bfloat16 if model.bfloat16_enabled() else None
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype is not None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            stack_out = model(src_input, tgt_input)
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())
        
            output = model_without_ddp.generate(stack_out, max_new_tokens=100, num_beams=4)
            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])

    tokenizer = model_without_ddp.mt5_tokenizer
    padding_value = tokenizer.eos_token_id
    pad_tensor = torch.ones(150 - len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0], pad_tensor.long()), dim=0)
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
            
    if args.dataset == 'CSL_News':
        tgt_pres = [' '.join(list(r.replace(" ", "").replace("\n", ""))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ",").replace("？", "?").replace(" ", ""))) for r in tgt_refs]

    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    for k, v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)

    metric_logger.synchronize_between_processes()
    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))
    
    if args.eval:
        with open(Path(args.output_dir) / 'tmp_pres.txt', 'w') as f:
            for pres in tgt_pres:
                f.write(pres + '\n')
        with open(Path(args.output_dir) / 'tmp_refs.txt', 'w') as f:
            for ref in tgt_refs:
                f.write(ref + '\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    # Ensure distributed training is disabled.
    args.distributed = False
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
