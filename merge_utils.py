import torch
import numpy as np
from model_utils import (
    load_base_model, 
    load_finetune_model, 
    save_base_model
)

class MergeTools():
    def __init__(self, merge_config):
        self.merge_method = merge_config.pop('merge_method', 'linear')
        self.base_model_name = merge_config.pop('base_model_name')
        self.base_model_ckpt_path = merge_config.pop('base_model_ckpt_path', '')
        self.base_model = load_base_model(self.base_model_name, self.base_model_ckpt_path)
        finetune_model_paths = merge_config.pop('finetune_model_paths')
        self.finetune_model_name = merge_config.pop('finetune_model_names', [])
        self.finetune_model = load_finetune_model(finetune_model_paths)
        self.check_model_compatibility()
        self.task_vectors = self.calculate_task_vectors()
        self.merge_config = merge_config
        self.NUM_LAYERS = 12

    def check_model_compatibility(self):
        base_state_dict = self.base_model
        for idx, finetune_state_dict in enumerate(self.finetune_model):
            # Check if keys are the same
            if finetune_state_dict.keys() != base_state_dict.keys():
                print(f"Finetune model at index {idx} has different keys.")
                continue
            # Check if the shape of all values are the same
            shape_mismatch = False
            for key in base_state_dict.keys():
                if key.startswith('model.final_proj') or key.startswith('model.label_embs_concat'):
                    continue
                if finetune_state_dict[key].shape != base_state_dict[key].shape:
                    print(f"Shape mismatch at key '{key}' for finetune model at index {idx}")
                    shape_mismatch = True
                    break
            if not shape_mismatch:
                print(f"Model at index {idx} is compatible.")
            else:
                print(f"Model at index {idx} has shape mismatches in state_dict.")
    
    def save_merge(self, new_params, output_dir, config):
        save_base_model(self.base_model_name, output_dir, new_params, config)

    def calculate_task_vectors(self):
        base_model = self.base_model
        finetune_models = self.finetune_model
        task_vectors = []
        for finetune_model in finetune_models:
            task_vector = {}
            for key in base_model.keys():
                if key.startswith('model.final_proj') or key.startswith('model.label_embs_concat'):
                    task_vector[key] = torch.zeros_like(base_model[key])
                    continue
                task_vector[key] = finetune_model[key] - base_model[key]
            task_vectors.append(task_vector)
        return task_vectors

    def merge(self):
        merge_function = getattr(self, self.merge_method, None)
        if callable(merge_function):
            return merge_function()
        else:
            raise AttributeError(f"Merge method '{self.merge_method}' not found in MergeTools.")
    
    def linear(self):
        alpha = self.merge_config.get('alpha', 0.25)
        new_params = {}
        for key in self.base_model.keys():
            base_param = self.base_model[key]
            tv_sum = sum(tv[key] for tv in self.task_vectors)/len(self.task_vectors)
            new_param = base_param + alpha * tv_sum
            new_params[key] = new_param
        return new_params