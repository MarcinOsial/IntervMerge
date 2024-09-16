#!/bin/bash

set -e
set -x

export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT=project_name
export WANDB_ENTITY=entity_name
export TMPDIR=/path/to/tmp/directory
export CUDA_VISIBLE_DEVICES="0"

group='tw_adamerg'
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
model_hidden_size=768 #VITL14
# VIT-B-32 768
# VIT-B-16 768
# VIT-L-14 1024

prior=0.3

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
    --train_lambdas \
    --use_lambda_custom    
