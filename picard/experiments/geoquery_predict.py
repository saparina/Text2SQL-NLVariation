import os
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Eval with official metrics.')
parser.add_argument('--model_name_or_path', type=str, help='Path to the model')
parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
parser.add_argument('--use_picard', action='store_true', help='Use Picard')
parser.add_argument('--data_dir', type=str, default="../data/geoquery_query_split/", help='Path to data')
args = parser.parse_args()

print(f'Predicting from {args.model_name_or_path}')

# Define base directories
base_config_dir = './configs/base_configs/'
config_root_dir = './configs/geoquery/'
eval_root_dir = './eval/geoquery/'

# Specific directories for this experiment
config_dir = os.path.join(config_root_dir, args.experiment_name)
output_dir = os.path.join(eval_root_dir, args.experiment_name)

# Ensure the existence of necessary directories
os.makedirs(config_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Initialize the content of the config file
config_file_content = ''

# Load and update the config JSON
config = 'eval_geoquery.json'
config_path = os.path.join(base_config_dir, config)
with open(config_path) as f:
    config_json = json.load(f)

config_json['model_name_or_path'] = args.model_name_or_path
config_json['output_dir'] = os.path.join(output_dir, 'geoquery')
config_json['data_dir'] = args.data_dir
config_json["test_sections"] = ["test"]

if args.use_picard:
    config_json["use_picard"] = True
    config_json["launch_picard"] = True
    config_json["picard_mode"] = "parse_with_guards"
    config_json["picard_schedule"] = "incremental"
    config_json["num_beams"] = 4
    config_json["picard_max_tokens_to_check"] = 2

# Save the updated config
new_config_path = os.path.join(config_dir, 'eval_geoquery.json')
with open(new_config_path, 'w') as f:
    json.dump(config_json, f, indent=4)

config_file_content += f'echo {new_config_path}\n'
config_file_content += f"PYTHONPATH='.' python seq2seq/run_seq2seq.py {new_config_path}\n"

config_file_content += 'echo "Evaluation is done!"'

# Write the commands to a shell script file
shell_script_path = os.path.join(config_root_dir, f'eval_geoquery_{args.experiment_name}.sh')
with open(shell_script_path, 'w') as f:
    f.write(config_file_content)

print()
print(f'sh {shell_script_path}')
