import torch
import yaml
import os
import sys
import argparse
from merge_utils import MergeTools

def main(config_path):
    # Load YAML configuration for merging
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        save_config = config.copy()
    output_dir = config.pop('output_dir')
    # Merge tools
    merge_tools = MergeTools(config)
    # Merge 
    new_params = merge_tools.merge()
    # Save model
    merge_tools.save_merge(new_params, output_dir, save_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and merge s3prl model")
    parser.add_argument('-c', '--config_path', type=str, help='Path to the YAML configuration file for merging')
    args = parser.parse_args()
    main(**vars(args))