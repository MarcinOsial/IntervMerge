#!/bin/bash

# Set environment variables
# export WANDB__SERVICE_WAIT=300
# export WANDB_PROJECT=project_name
# export WANDB_ENTITY=entity_name
# export TMPDIR=/path/to/tmp/directory
export WANDB_API_KEY=66d9e2b16753e25dd022a685b96fadf363a8f58b
#"f61fe6de67dc18515ebe11ca944faaa2ccdd11e1"
export WANDB__SERVICE_WAIT=300
export WANDB_PROJECT="intervmerge"
export WANDB_ENTITY="osialm"
export TMPDIR="/raid/NFS_SHARE/home/marcin.osial/AdaMerging/tmp/"
export CUDA_VISIBLE_DEVICES="0"

# Set experiment parameters
group='lw_adamerging'
name="experiment_name"
model='ViT-B-32' 
batch_size=128
data_location='/home/marcin.osial/AdaMerging/data'
save_checkpoints='checkpoints'
logs='logs'
device='cuda:0'

exam_datasets="SUN397 Cars RESISC45 EuroSAT SVHN GTSRB MNIST DTD"
iterations=500
reft_position=0 # class token is at '0' position
start_layer=11
num_layer=12
model_hidden_size=768
# model_hidden_size=64

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
    --layerwise \
    --use_lambda_custom \
    --reft_position=$reft_position \
    --num_layer=$num_layer \
    --start_layer=$start_layer \
    --seed 42 \
    --low_rank_dimension 1 \
    --model_hidden_size=$model_hidden_size \
    --save_model_flag \
    --use_intervention \
    --train_intervention \
    --train_lambdas

# --intervention_type 'miniInterv_best_efficient'