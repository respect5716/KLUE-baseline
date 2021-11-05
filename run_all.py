import os
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()

OUTPUT_DIR = 'klue_output'
DATA_DIR = 'data/klue_benchmark'
VERSION = 'v1.1'
GPUS = 0
NUM_WORKERS = 4

def read_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return config

def make_command(model, task, task_config):
    cmd = 'python run_klue.py train'
    cmd += f' --output_dir {OUTPUT_DIR}'
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
    config = read_config()
    tasks = ['ynat', 'klue-nli', 'klue-ner', 'klue-re', 'klue-dp', 'klue-mrc', 'wos']
    for task in tasks:
        command = make_command(args.model, task, config[task])
        os.system(command)


if __name__ == '__main__':
    main(args)