#!/bin/bash
source activate vim_env
cd projects/Vim/vim;

# CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 64 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 25 --data-path /home/filippo/datasets/tiny_imagenet/train --output_dir ./output/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
CUDA_VISIBLE_DEVICES=1 python main.py --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 64 --drop-path 0.05 --weight-decay 0.05 --lr 1e-3 --num_workers 25 --data-path /home/filippo/datasets/tiny_imagenet --output_dir ./tmp/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp
