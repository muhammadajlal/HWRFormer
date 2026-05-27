"""Run training in a 5-fold cross-validation fashion.

Generates one temporary per-fold config from a `idx_fold: -1` config and runs
`main.py` on each fold sequentially, then removes the temporary directory.

    python train_cv.py -c configs/train.yaml
    python train_cv.py -c configs/train.yaml -m main.py

Each fold writes to `<dir_work>/<fold>/`; aggregate the folds afterwards with
`python evaluate.py -c configs/train.yaml`.
"""

import argparse
import json
import os
import sys

import yaml

from hwrformer.utils import expand_cfg_paths


def train_cv(cfgs: dict, path_main: str) -> None:
    """Train a model across all cross-validation folds.

    Args:
        cfgs (dict): Training configuration (must have ``idx_fold: -1``).
        path_main (str): Path to the per-fold training script (``main.py``).
    """
    # Number of folds is recorded in the dataset's train.json `info` block.
    dir_dataset = expand_cfg_paths(cfgs['dir_dataset'])
    with open(os.path.join(dir_dataset, 'train.json'), 'r') as f:
        num_fold = json.load(f)['info']['num_fold']

    dir_temp = f'temp_{os.path.basename(cfgs["dir_work"])}'
    os.makedirs(dir_temp, exist_ok=True)

    commands = []
    for i in range(num_fold):
        cfgs['idx_fold'] = i
        path_temp = os.path.join(dir_temp, f'f{i}.yaml')
        with open(path_temp, 'w') as f:
            yaml.safe_dump(cfgs, f)
        commands.append(f'{sys.executable} {path_main} -c {path_temp}')

    # Run folds sequentially; on success, remove the temporary configs.
    command = ' && '.join(commands) + f' && rm -rf {dir_temp}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition training with cross validation.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to the YAML file of configuration.'
    )
    parser.add_argument(
        '-m', '--main', default='main.py',
        help='Path to the per-fold training script.',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)

    assert cfgs['idx_fold'] == -1, (
        'train_cv.py expects idx_fold: -1 (all folds). For a single fold, set '
        'idx_fold to 0..N-1 and run main.py directly.'
    )

    train_cv(cfgs, args.main)
