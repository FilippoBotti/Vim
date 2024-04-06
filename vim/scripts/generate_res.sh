cd projects/Vim/vim;

EVAL_FLAGS="--eval --style_transfer --no_amp
            --resume /home/filippo/projects/Vim/vim/output/cross-vim_e2s/checkpoint.pth
            --model vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 
            --data-path /home/filippo/datasets/edges2shoes --data-set EDGES2SHOES --input-size 256 
            --output_dir /home/filippo/projects/Vim/vim/output/cross-vim_e2s --batch 32"

CUDA_VISIBLE_DEVICES=0 python main.py $EVAL_FLAGS