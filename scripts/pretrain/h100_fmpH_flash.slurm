#!/bin/bash

#SBATCH --job-name="flsH"
#SBATCH -A vfj@h100                  
#SBATCH -C h100                     
#SBATCH --gres=gpu:4
#SBATCH --nodes=10                  
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=24   
#SBATCH --hint=nomultithread    
#SBATCH --qos=qos_gpu_h100-t3                                       

#SBATCH --time=20:00:00            
#SBATCH --output=./logs/pretrain/OUTPUT_pretrain_fmpH_flash.out    
#SBATCH --error=./logs/pretrain/ERROR_pretrain_fmpH_flash.out                                      

module purge
module load arch/h100
module load cuda/12.1.0 
module load cudnn/9.2.0.82-cuda 
module load gcc/11.3.1
module load miniforge/24.9.0 

conda activate VideoMAE

OUTPUT_DIR='./output/cross_reconstruct/fmp/pretrain/flash_vox_celeb2_pretrain_HUGE_K32_patch16_224_frame_16x4_tube_mask_ratio_0.9_e201_lr8e-5_tau0.17_aug_droppath0.1'
DATA_PATH='./datasets/dataSpecCSV_combined/vox-celeb-dataset_video_specs_updated_only_inliers.csv'

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

set -x

srun python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --model pretrain_videomae_huge_patch16_224 \
        --decoder_depth 12 \
        --batch_size 64 \
        --num_workers 24 \
        --num_frames 16 \
        --sampling_rate 4 \
        --lr 8e-5 \
        --drop_path 0.1 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 20 \
        --epochs 201 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --add_fmp_attention \
        --fmp_use_residual \
        --fmp_num_last_layers 32 \
        --use_fmp_flashlite

