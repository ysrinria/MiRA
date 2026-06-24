#!/bin/bash

#SBATCH --job-name="fmpB"
#SBATCH -A ntk@h100                  
#SBATCH -C h100                     
#SBATCH --gres=gpu:4
#SBATCH --nodes=8                  
#SBATCH --ntasks-per-node=4     
#SBATCH --cpus-per-task=24   
#SBATCH --hint=nomultithread    
#SBATCH --qos=qos_gpu_h100-t3                                       

#SBATCH --time=20:00:00            
#SBATCH --output=JZ_OUTPUT_pretrain_fmpB.out    
#SBATCH --error=JZ_ERROR_pretrain_fmpB.out                                      

module purge
module load arch/h100
module load cuda/12.1.0 
module load cudnn/9.2.0.82-cuda 
module load gcc/11.3.1
module load miniforge/24.9.0 

conda activate VideoMAE

OUTPUT_DIR='./output/cross_reconstruct/fmp/pretrain/vox_celeb2_pretrain_BASE_K12_patch16_224_frame_16x4_tube_mask_ratio_0.9_e301'
DATA_PATH='./datasets/dataSpecCSV_combined/vox-celeb-dataset_video_specs_updated_only_inliers.csv'

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

set -x

srun python -u run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --no_aug_paired_input \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 128 \
        --num_workers 16 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 20 \
        --save_ckpt_freq 20 \
        --epochs 301 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --no_aug_paired_input \
        --use_auto_reconstruct \
        --add_fmp_attention \
        --fmp_use_residual \
        --fmp_num_last_layers 12 \