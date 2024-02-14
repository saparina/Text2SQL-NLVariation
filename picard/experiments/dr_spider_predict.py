import os
import json
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate config files for Dr.Spider evaluation.')
parser.add_argument('--model_name_or_path', type=str, help='Model name or path to the checkpoint directory')
parser.add_argument('--parts', type=str, help='Parts of eval set: NLQ, SQL, DB, all')
parser.add_argument('--experiment_name', type=str, help='Name of the experiment')
parser.add_argument('--use_picard', action='store_true', help='Use Picard')
parser.add_argument('--data_dir', type=str, default="../data/diagnostic-robustness-text-to-sql/data/", help='Path to data')
args = parser.parse_args()

print(f'Generating prediction config for {args.model_name_or_path}')

# Define base directories
base_config_dir = './configs/base_configs/'
config_root_dir = './configs/dr_spider/'
eval_root_dir = './eval/dr_spider/'

for dir_path in [config_root_dir, eval_root_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Specific directories for this experiment
config_dir = os.path.join(config_root_dir, args.experiment_name)
output_dir = os.path.join(eval_root_dir, args.experiment_name)
pre_output_dir = os.path.join(output_dir, 'pre')
post_output_dir = os.path.join(output_dir, 'post')

for dir_path in [config_dir, output_dir, pre_output_dir, post_output_dir]:
    os.makedirs(dir_path, exist_ok=True)

perturbation_names = []

if args.parts in ['NLQ', 'all']:
    perturbation_names += [
        'NLQ_keyword_carrier', 'NLQ_keyword_synonym', 'NLQ_column_synonym', 'NLQ_column_carrier', 'NLQ_column_attribute', 'NLQ_column_value',
        'NLQ_value_synonym', 'NLQ_multitype', 'NLQ_others'
    ]
if args.parts in ['SQL', 'all']:
    perturbation_names += [
        'SQL_DB_text', 'SQL_comparison', 'SQL_DB_number', 'SQL_NonDB_number', 'SQL_sort_order'
    ]
if args.parts in ['DB', 'all']:
    perturbation_names += [
        'DB_schema_abbreviation', 'DB_DBcontent_equivalence', 'DB_schema_synonym'
    ]

configs = [f'eval_dr_spider.json', f'eval_dr_spider_original.json']
config_file_content = ''

# Generate configs for additional perturbations
for name in perturbation_names:
    for idx, config in enumerate(configs):
        config_path = os.path.join(base_config_dir, config)
        with open(config_path) as f:
            config_json = json.load(f)

        # Replace data, output, and db directory names
        config_json['data_dir'] = os.path.join(args.data_dir, name)
        config_json['output_dir'] = os.path.join(post_output_dir if idx == 0 else pre_output_dir, name)
        config_json['model_name_or_path'] = args.model_name_or_path
        if args.use_picard:
            config_json["use_picard"] = True
            config_json["launch_picard"] = True
            config_json["picard_mode"] = "parse_with_guards"
            config_json["picard_schedule"] = "incremental"
            config_json["num_beams"] = 4
            config_json["picard_max_tokens_to_check"] = 2

        new_config_path = os.path.join(config_dir, config.replace('.json', f'_{name}.json'))
        with open(new_config_path, 'w') as f:
            json.dump(config_json, f, indent=4)

        config_file_content += f'echo {new_config_path}\n'
        config_file_content += f"PYTHONPATH='.' python seq2seq/run_seq2seq.py {new_config_path}\n"

config_file_content += 'echo "Evaluation is done!"'

# Write commands to a shell script file
os.makedirs('configs/dr_spider/', exist_ok=True)
with open(f'configs/dr_spider/eval_dr_spider_{args.experiment_name}.sh', 'w') as f:
    f.write(config_file_content)

print()
print(f'sh configs/dr_spider/eval_dr_spider_{args.experiment_name}.sh')