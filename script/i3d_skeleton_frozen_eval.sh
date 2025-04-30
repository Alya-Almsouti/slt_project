output_dir=out/i3d_skeleton_frozen_eval
ckpt_path=out/i3d_skeleton_frozen/best_checkpoint.pth
deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 1 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset CSL_News \
   --rgb_support \
   --vid_extractor resnet \
   --skeleton_support \
   --skeleton_extractor unisign \
   --freeze_vision_encoder \
   --eval \
   --finetune $ckpt_path \
   --show_predictions