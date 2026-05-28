import os
import sys
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import yaml
from src.training.trainer import Trainer


os.environ["USE_LIBUV"] = "0"

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-config",
        default="configs/train_metadrive.yaml",
        help="Path to training config yaml (e.g. configs/train.yaml)",
    )

    args = parser.parse_args()

    train_path = ROOT_DIR / args.train_config
    if not train_path.exists():
        print(f"ERROR: train config not found: {train_path}", file=sys.stderr)
        raise SystemExit(2)

    config = load_config(train_path)
    if config is None:
        print(f"ERROR: train config is empty: {train_path}", file=sys.stderr)
        raise SystemExit(2)

    print('\033[34mUsing following config:\033[0m')
    for key, val in config.items():
        prefix = '| '
        if isinstance(val, dict):
            print(prefix + '\033[33m' + str(key) + '\033[0m' + ':')
            prefix += '  - '
            for item_key, item_val in val.items():
                print(prefix + '\033[33m' + str(item_key) + '\033[0m' + ': ' + str(item_val))
        else: 
            print(prefix + '\033[33m' + str(key) + '\033[0m' + ': ' + str(val))
    print('')

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn")
    main()