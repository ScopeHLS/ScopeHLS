import argparse
import yaml

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=".config/test.yaml", help='Path to YAML config file')
parser.add_argument('--main_mode', type=str, default='inference', choices=['train', 'inference'], help='Main mode of operation')
args = parser.parse_args()

if args.config:
    yaml_config = load_yaml_config(args.config)
    for key, value in yaml_config.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

CONFIG = args