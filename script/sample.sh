output_dir=out/i3d_skeleton_frozen_eval
ckpt_path=checkpoints/i3d/best_checkpoint.pth
deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 1 \
   --gradient-accumulation-steps 1 \
   --epochs 1 \
   --opt AdamW \
   --lr 5e-5 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset sample \
   --rgb_support \
   --vid_extractor i3d \
   --skeleton_support \
   --skeleton_extractor unisign \
   --freeze_vision_encoder \
   --eval \
   --finetune $ckpt_path \
   --show_predictions