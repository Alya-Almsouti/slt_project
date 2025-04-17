
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from models import Base_Model, get_requires_grad_dict
import utils
from datasets import VidText_Dataset
import wandb
import os
import time
import argparse
import json
import datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from transformers import get_scheduler
from SLRT_metrics import translation_performance
from config import *

def main(args):
    utils.set_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            project="Uni-Sign",
            name=args.run_name if hasattr(args, 'run_name') else 'FullClips',
            config=vars(args)
        )

    # Prepare training and development datasets.
    train_data = VidText_Dataset(path=train_label_paths[args.dataset], args=args)
    print(f"Train labels from: {train_label_paths[args.dataset]}")
    print(f"Train dataset: {train_data}, Length: {len(train_data)}")

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

    

    print("Creating model...")
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    model = Base_Model(args=args)
    model = model.cuda().to(dtype=torch.bfloat16)
    model.train()    

    # Convert all trainable parameters to bf16 (if needed)
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
    
    n_parameters = utils.count_parameters_in_MB(model)
    print(f'Number of parameters: {n_parameters}M')
    
    optimizer = create_optimizer(args, model)

    if args.quick_break <= 0:
        args.quick_break = len(train_dataloader)

    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0

    
    if args.eval:
        checkpoint_path = Path('out/base/best_checkpoint.pth')
        print(f"✅ Loading Best Checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(checkpoint['model'])

        print("📄 test result")
        test_data = VidText_Dataset(path=test_label_paths[args.dataset], 
                                args=args)
        test_dataloader = DataLoader(test_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers, 
                                    collate_fn=dev_data.collate_fn)
        test_stats = evaluate(args, test_dataloader, model)
        return 
    

    print(f"Start training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):

        # Train for one epoch.
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)
    
        # Save the model
        if args.output_dir:
            checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
            torch.save({'model': get_requires_grad_dict(model)}, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, model)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")


        wandb.log({
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"val/{k}": v for k, v in test_stats.items()},
            })

        if max_accuracy < test_stats["bleu4"]:
            max_accuracy = test_stats["bleu4"]
            if args.output_dir:
                checkpoint_path = output_dir / f'checkpoint_{epoch}.pth'
                torch.save({'model': get_requires_grad_dict(model)}, checkpoint_path)
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.use_wandb:
        wandb.finish() 
    print('ALL good untill now')

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    """
    Run one training epoch.
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}/{args.epochs}]'
    print_freq = 100
    optimizer.zero_grad()

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # Convert input tensors to bfloat16 before passing them to model
        for key in src_input.keys():
            if isinstance(src_input[key], torch.Tensor):
                src_input[key] = src_input[key].to(dtype=torch.bfloat16).cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):  
            stack_out = model(src_input, tgt_input)  

        total_loss = stack_out['loss']
        total_loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if step % print_freq == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        #break

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def evaluate(args, data_loader, model):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):

            # Ensure src_input is converted to bfloat16
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(dtype=torch.bfloat16).cuda()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                stack_out = model(src_input, tgt_input)
                total_loss = stack_out['loss']
                metric_logger.update(loss=total_loss.item())

                # Ensure model.generate() runs in autocast mode
                output = model.generate(stack_out, 
                                        max_new_tokens=100, 
                                        num_beams=4)

            for i in range(len(output)):
                tgt_pres.append(output[i])
                tgt_refs.append(tgt_input['gt_sentence'][i])


    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    pad_tensor = torch.ones(150-len(tgt_pres[0])).cuda() * padding_value
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)

    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
            
    if args.dataset == 'CSL_News':
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
    for k,v in bleu_dict.items():
        metric_logger.meters[k].update(v)
    metric_logger.meters['rouge'].update(rouge_score)


    print('* BLEU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.bleu4, losses=metric_logger.loss))
    
    if args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

import os



if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    print(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)