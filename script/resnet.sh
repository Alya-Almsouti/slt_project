output_dir=out/resent_eval

deepspeed --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 2 \
   --gradient-accumulation-steps 4 \
   --epochs 20 \
   --opt AdamW \
   --lr 5e-5 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset CSL_News \
   --rgb_support \
   --vid_extractor resnet \
   --eval
