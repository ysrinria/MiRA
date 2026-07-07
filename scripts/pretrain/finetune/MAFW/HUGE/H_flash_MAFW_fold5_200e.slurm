#!/bin/bash

#SBATCH --job-name="MA5Hf2e1"
#SBATCH -A vfj@h100 
#SBATCH -C h100 
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24 
#SBATCH --hint=nomultithread 
#SBATCH --qos=qos_gpu_h100-t3 

#SBATCH --time=20:00:00 
#SBATCH --output=./logs/MAFW/HUGEflash/OUTPUT_finetune_MAFW_fold5_HUGEflash_pre200e_lr1e-3_instance.out
#SBATCH --error=./logs/MAFW/HUGEflash/ERROR_finetune_MAFW_fold5_HUGEflash_pre200e_lr1e-3_instance.out


module purge
module load arch/h100
module load cuda/12.1.0
module load cudnn/9.2.0.82-cuda
module load gcc/11.3.1
module load miniforge/24.9.0

conda activate VideoMAE


export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_ext_${SLURM_JOB_ID}
export DS_BUILD_CACHE_DIR=/tmp/$USER/ds_cache_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR" "$DS_BUILD_CACHE_DIR"

export OMP_NUM_THREADS=1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=$((12000 + RANDOM % 20000))


OUTPUT_DIR='./output/cross_reconstruct/fmp/main_results/MAFW/fold5/HUGEflash/finetune_MAFW_fold5_HUGEflash_pre200e_lr1e-3_instance'
DATA_PATH="./datasets/dataSpecCSV_combined/MAFW_video_specs_updated_fold_5.csv"
MODEL_PATH='./output/cross_reconstruct/fmp/pretrain/flash_vox_celeb2_pretrain_HUGE_K32_patch16_224_frame_16x4_tube_mask_ratio_0.9_e201_lr8e-5_tau0.17_aug_droppath0.1/checkpoint-199.pth'

set -x

srun --mpi=none --kill-on-bad-exit=1 \
  python -u run_class_finetuning.py \
        --model vit_huge_patch16_224 \
        --data_set MAFW_crop \
        --nb_classes 11 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 24 \
        --batch_size 16 \
        --num_sample 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 1e-3 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 100 \
        --warmup_epochs 10 \
        --dist_eval \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --enable_deepspeed \
        --add_fmp_attention \
        --fmp_use_residual \
        --fmp_num_last_layers 32 \
        --use_fmp_flashlite \
        --stats_mode instance \



