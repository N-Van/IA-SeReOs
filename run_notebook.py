import papermill as pm
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define your parameters
params = {
    'batch_size': 32,
    'period_size': 8,
    'weight_decay': 0.1,
    'epsilon': 10**(-8),
    'epochs_num': 7,
    'learning_rate': 10**(-3),
    'n_channel': 3,
    'class_num': 3,
    'air_threshold': 0.9,
    'n_channels': 3,
    'step': 1,
    'filename_prefix': "os_long",
    'stride': 32,  
    'train_size': 0.7,
    'output': 128,
    'dropout_rate': 0.5,
    'start_index': 0,
    'end_index': 30
}

# List of parameter sets
param_sets = [
    {'epochs_num': 5},
    {'epochs_num': 6},
    {'epochs_num': 9},
    {'epochs_num': 10},
    # Add more parameter sets as needed
]

# Path to your notebook
notebook_path = 'C:/Users/n.vanderesse/Desktop/STAGE_Nolan/Git/IA-SeReOs/training_adapted.ipynb'

# Define the output directory within your project
output_dir = 'C:/Users/n.vanderesse/Desktop/STAGE_Nolan/Git/IA-SeReOs/training_output'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each parameter set and execute the notebook
for i, extra_params in enumerate(param_sets):
    output_notebook = os.path.join(output_dir, f'output_notebook_{params["filename_prefix"]}_{i}.ipynb')
    merged_params = {**params, **extra_params}
    logger.info(f"Executing notebook with parameters: {merged_params}")
    try:
        pm.execute_notebook(notebook_path, output_notebook, parameters=merged_params)
    except Exception as e:
        logger.error(f"Error executing notebook with parameters {merged_params}: {e}")

logger.info("All notebooks have been executed.")
