#!/bin/bash
cd projects/Vim/vim;

TRAIN_FLAGS="--model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2
                    --drop-path 0.05 --weight-decay 0.05 --lr 1e-4 --drop-path 0.0  --num_workers 25 --no_amp
                    --data-path /home/filippo/datasets/edges2shoes --output_dir ./output/vgg_loss
                    --epochs 100 --finetune /home/filippo/projects/Vim/vim/output/vgg_loss/checkpoint.pth
                    --vgg /home/filippo/checkpoints/sty-try/vgg_normalised.pth
                     --data-set EDGES2SHOES --input-size 256 --style_transfer --batch-size 16"
CUDA_VISIBLE_DEVICES=0 python main.py $TRAIN_FLAGS
                                        
