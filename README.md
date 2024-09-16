## Datasets and Checkpoints
Refer to dataset processing in the [task_vectors](https://github.com/mlfoundations/task_vectors).
To download the datasets, refer to the [Surgery](https://github.com/EnnengYang/RepresentationSurgery) repository:


### Train

**If you want to train IntervMerge, run the appropriate .sh file:**

Before running the scripts, make sure to update the following paths, among others, in each .sh file:

1. Update the working directory:
   ```
   cd /path/to/your/project/directory
   ```

2. Update the Conda environment activation:
   ```
   source /path/to/your/miniconda3/bin/activate /path/to/your/conda/env
   ```

3. Adjust the following variables as needed:
   ```
   data_location='path/to/your/data'
   save_checkpoints='path/to/save/checkpoints'
   logs='path/to/save/logs'
   ```

4. If using Weights & Biases (wandb), update the project and entity:
   ```
   export WANDB_PROJECT=your_project_name
   export WANDB_ENTITY=your_entity_name
   ```

After updating all necessary configurations, you can run the appropriate .sh file for your chosen merging method:

- For Layer-wise AdaMerging: `bash src/intervmerge_adamerging.sh`
- For Task-wise AdaMerging: `bash src/intervmerge_tw_adamerging.sh`
- For Task Arithmetic: `bash src/intervmerge_task_artimetic.sh`
- For Ties Merging: `bash src/intervmerge_ties_merging.sh`
- For Average Weight: `bash src/intervmerge_avg_weights.sh`

Choose the appropriate script based on the merging method you want to use.


### Detailed Configuration

For more detailed configuration options, please refer to the `config.py` file. 