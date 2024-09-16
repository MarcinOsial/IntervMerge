#!/bin/bash

# Set environment variables
export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=project_name
export WANDB_ENTITY=entity_name
export TMPDIR=/path/to/tmp/directory
export CUDA_VISIBLE_DEVICES="0"

# Set experiment parameters
group='avg_weights'
name="experiment_name"
model='ViT-B-32' 
batch_size=128
data_location='data'
save_checkpoints='checkpoints'
logs='logs'
device='cuda:0'
exam_datasets="SUN397 Cars RESISC45 EuroSAT SVHN GTSRB MNIST DTD"
iterations=500

reft_position=0
start_layer=11
num_layer=12
model_hidden_size=768
prior=0.125

# Run the main script
python src/main_intervmerge.py \
    --iterations $iterations \
    --name $name \
    --group $group \
    --model $model \
    --batch_size $batch_size \
    --data_location $data_location \
    --save_checkpoints $save_checkpoints \
    --logs $logs \
    --device $device \
    --exam_datasets $exam_datasets \
    --wandb \
    --deterministic \
    --use_intervention \
    --train_intervention \
    --reft_position=$reft_position \
    --num_layer=$num_layer \
    --start_layer=$start_layer \
    --seed 1 \
    --low_rank_dimension 1 \
    --model_hidden_size=$model_hidden_size \
    --taskwise \
    --prior=$prior \
    --save_model_flag
