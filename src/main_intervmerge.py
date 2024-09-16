import os
import sys
import random
import torch
import tqdm
from datetime import datetime
import wandb
torch.set_printoptions(threshold=10_000)
from models.model import AdaMerging
from datasets.common import get_dataloader_shuffle, maybe_dictionarize
from datasets.registry import get_dataset
from utils.model_utils import ModelWrapper
from task_vectors import TaskVector
import torch
from ties_merging_utils import *
from matplotlib import pyplot as plt
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle
from eval import eval_single_dataset_preprocess_mapping_head
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from configs.config import Config
import uuid
import subprocess, datetime
import torch.nn as nn

config = Config.get_instance()

def git_commit_and_push(config):
    # Generate a unique commit message with the current timestamp
    commit_message = f"Auto-commit: Script run at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


    try:
        # Add changes to the staging area
        subprocess.run(["git", "add", "."], check=True)
        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        # Push the changes
        subprocess.run(["git", "push"], check=True)
        print("Changes committed and pushed to Git repository.")
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")

os.environ["WANDB_SILENT"] = "true"
git_commit_and_push(config)
if config.wandb:
    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        config=config,
        name=config.name,
        dir=config.wandb_logs,
        group=config.group,
    )

if config.pruned:
    ft_checks = [torch.load(os.path.join(config.save_checkpoints, dataset_name, 'finetuned.pt')).state_dict() for dataset_name in config.exam_datasets]
    ptm_check = torch.load(config.pretrained_checkpoint).state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])
    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack([state_dict_to_vector(check, remove_keys) for check in ft_checks])
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)
    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check)
    assert all([check_state_dicts_equal(vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i])for i in range(len(ft_checks))])
    K = 20
    merge_func = "dis-sum"
    selected_entries, merged_tv = ties_merging_split(tv_flat_checks, reset_thresh=K, merge_func=merge_func)
    task_vectors = []
    for vector_ in selected_entries:

        t_state_dict = vector_to_state_dict(vector_, ptm_check, remove_keys=remove_keys)

        ref_model = torch.load(config.pretrained_checkpoint)

        ref_model.load_state_dict(t_state_dict, strict=False)

        task_vectors.append(ref_model.state_dict())
else:
    task_vectors = [TaskVector(config.pretrained_checkpoint, os.path.join(config.save_checkpoints, dataset_name, 'finetuned.pt')) for dataset_name in config.exam_datasets]


pretrained_model = torch.load(config.pretrained_checkpoint, map_location='cuda:0')
pretrained_model_dic = pretrained_model.state_dict()
model = ModelWrapper(pretrained_model, config)
model = model.to(config.device)

paramslist = []
paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in pretrained_model_dic.items())] # pretrain

if config.pruned:
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.items())  for i, tv in enumerate(task_vectors)] # task vectors
else:
    paramslist += [tuple(v.detach().requires_grad_().cpu() for _, v in tv.vector.items())  for i, tv in enumerate(task_vectors)] # task vectors

adamerging_mtl_model = AdaMerging(paramslist, model)
params_dict = adamerging_mtl_model.collect_trainable_params()


optimizer_groups = []
for key in params_dict.keys():
    if "position" == key:
        continue
    optimizer_groups.append({'params': params_dict[key], 'lr': 1e-3})

optimizer = torch.optim.Adam(optimizer_groups, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.)

config.count = 0
student_Total_ACC = 0.
teacher_Total_ACC = 0.

for iteration in tqdm.tqdm(range(config.iterations + 1)):
    print("Iteration", iteration)
    if ((iteration) % config.eval_iterations) == 0 and iteration != 0 or iteration in [100, 500]:
        config.phase = "eval"
        student_Total_ACC = 0.
        teacher_Total_ACC = 0.

        for ddi, dataset_name in enumerate(config.exam_datasets):
            image_encoder, interventions = adamerging_mtl_model.get_image_encoder()
            classification_head = adamerging_mtl_model.get_classification_head(dataset_name)

            config.current_dataset = dataset_name
            config.current_model = "student"

            
            student_metrics = eval_single_dataset_preprocess_mapping_head(image_encoder, classification_head, dataset_name, config, interventions)
            student_Total_ACC += student_metrics['top1']

            wandb.log({f"acc_student_e_eval_"+str(dataset_name): student_metrics['top1'], "iteration": iteration})

        print("student_Total_ACC:", f"{student_Total_ACC / len(config.exam_datasets)}")

        adamerging_mtl_model.move_classification_head_to_cpu(dataset_name)
        wandb.log({f"acc_student_e_eval": student_Total_ACC / len(config.exam_datasets), "iteration": iteration})
        
    if iteration in [500] and config.save_model_flag:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        adamerging_mtl_model.save(save_directory=config.save_path, custom_filename=f"{config.name}_iter_{iteration}_{'_'.join(config.exam_datasets)}_{current_time}.pt", include_model=True)

    if config.iterations == iteration:
        break 

    config.phase = "train"
    accuracy_student_both = 0.0

    for dataset_name in config.exam_datasets:
        dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=config.data_location, batch_size=16)
        dataloader = get_dataloader_shuffle(dataset)
        config.current_dataset = dataset_name
        correct_predictions, total_samples = 0, 0
        teacher_correct_predictions = 0
        teacher_total_samples = 0

        adamerging_mtl_model.move_classification_head_to_gpu(dataset_name)

        for i, data in enumerate(dataloader):
            config.current_model = "student"
            accuracy_student = 0
            accuracy_teacher = 0
            torch.cuda.empty_cache()
            
            data = maybe_dictionarize(data)
            x = data['images'].to(config.device)
            labels = data['labels'].to(config.device)

            adamerging_mtl_model.train()
            outputs, feature = adamerging_mtl_model(x, dataset_name)

            predictions = outputs.argmax(dim=1, keepdim=True)
            correct_predictions += predictions.eq(labels.view_as(predictions)).sum().item()
            total_samples += labels.size(0)
            accuracy_student = correct_predictions / total_samples if total_samples > 0 else 0

            config.current_model = "teacher"

            if len(config.exam_datasets) == 8:
                if dataset_name == "SUN397":
                    finetuning_factors = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif dataset_name == "Cars":
                    finetuning_factors = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif dataset_name == "RESISC45":
                    finetuning_factors = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif dataset_name == "EuroSAT":
                    finetuning_factors = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
                elif dataset_name == "SVHN":
                    finetuning_factors = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                elif dataset_name == "GTSRB":
                    finetuning_factors = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
                elif dataset_name == "MNIST":
                    finetuning_factors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
                elif dataset_name == "DTD":
                    finetuning_factors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                else:
                    raise ValueError(f"Unknown dataset: {dataset_name}")

                if config.model == "ViT-L-14":
                    number_of_layers = 302
                elif config.model == "ViT-B-32":
                    number_of_layers = 158
                elif config.model == "ViT-B-16":
                    number_of_layers = 158

                teacher_custom_rlambdas = torch.cat([
                    torch.full((number_of_layers, 1), factor) for factor in finetuning_factors
                ], dim=1)

                if config.taskwise:
                    teacher_custom_rlambdas = torch.tensor(finetuning_factors).unsqueeze(0)

            current_learnable_rlambdas = adamerging_mtl_model.get_lambdas_raw()
            torch.set_printoptions(threshold=current_learnable_rlambdas.numel())
            # print("Current learnable rlambdas:",current_learnable_rlambdas)
            adamerging_mtl_model.set_lambdas_raw(teacher_custom_rlambdas)

            adamerging_mtl_model.eval()
            with torch.no_grad():
                teacher_outputs, feature_teacher = adamerging_mtl_model(x, dataset_name)

            teacher_predictions = teacher_outputs.argmax(dim=1, keepdim=True)


            teacher_correct_predictions += teacher_predictions.eq(labels.view_as(teacher_predictions)).sum().item()
            teacher_total_samples += labels.size(0)

            accuracy_teacher = teacher_correct_predictions / teacher_total_samples if teacher_total_samples > 0 else 0
            adamerging_mtl_model.set_lambdas_raw(current_learnable_rlambdas)
            adamerging_mtl_model.train()


            loss_func = torch.nn.L1Loss()
            loss = loss_func(feature, feature_teacher)
            wandb.log({"loss_mean": loss})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            config.count += 1
            if i > 0:
                break
        
    
        wandb.log({f"acc_student_e_train_{config.current_dataset}": accuracy_student, "iteration": iteration})
        adamerging_mtl_model.move_classification_head_to_cpu(config.current_dataset)
        accuracy_student_both += accuracy_student

    wandb.log({f"acc_student_e_train": accuracy_student_both / len(config.exam_datasets), "iteration": iteration})
    

wandb.finish(quiet=True)
