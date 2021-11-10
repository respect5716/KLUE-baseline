import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--tasks', type=str, default='all')
parser.add_argument('--config', type=str, default='config')
parser.add_argument('--output_dir', type=str, default='klue_output')
args = parser.parse_args()

DATA_DIR = 'data/klue_benchmark'
VERSION = 'v1.1'
GPUS = 0
NUM_WORKERS = 4

def read_config(config_fname):
    with open(f'{config_fname}.json', 'r') as f:
        config = json.load(f)
    return config

def make_command(model, task, task_config, output_dir):
    cmd = 'python run_klue.py train'
    cmd += f' --output_dir {output_dir}'
    cmd += f' --data_dir {DATA_DIR}/{task}-{VERSION}'
    cmd += f' --gpus {GPUS}'
    cmd += f' --num_workers {NUM_WORKERS}'
    cmd += f' --task {task}'
    cmd += f' --model_name_or_path {model}'

    for k, v in task_config.items():
        if type(v) == bool and v:
            cmd += f' --{k}'
        else:
            cmd += f' --{k} {v}'

    return cmd


def main(args):
    config = read_config(args.config)

    if args.tasks.lower() == 'all':
        tasks = ['ynat', 'klue-nli', 'klue-ner', 'klue-re', 'klue-dp', 'klue-mrc', 'wos']
    else:
        tasks = [t.lower() for t in args.tasks.split(',')]

    for task in tasks:
        command = make_command(args.model, task, config[task], args.output_dir)
        os.system(command)


if __name__ == '__main__':
    main(args)
