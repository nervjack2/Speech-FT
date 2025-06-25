import os 
import sys
import torch
import yaml
from s3prl.util.download import _urls_to_filepaths
from typing import List

def load_base_model(model_name: str, model_ckpt_path: str):
    function_name = f"load_{model_name}_model"
    if callable(eval(function_name)): 
        model = eval(function_name)(model_ckpt_path)
    else:
        raise NotImplementedError(f"Model hasn't been implemented yet.")
    return model

def load_hubert_model(ckpt):
    if not len(ckpt):
        ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=False)
    from s3prl.upstream.hubert.expert import UpstreamExpert as _UpstreamExpert
    model = _UpstreamExpert(ckpt, model_config=None)
    return model.state_dict()

def load_wav2vec2_model(ckpt):
    if not len(ckpt):
        ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_small.pt"
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=False)
    from s3prl.upstream.wav2vec2.expert import UpstreamExpert as _UpstreamExpert
    model = _UpstreamExpert(ckpt, model_config=None)
    return model.state_dict()

def load_decoar2_model(ckpt):
    if not len(ckpt):
        ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/checkpoint_decoar2.pt"
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=False)
    from s3prl.upstream.decoar2.expert import UpstreamExpert as _UpstreamExpert
    model = _UpstreamExpert(ckpt, model_config=None)
    return model.state_dict()

def load_wavlm_base_plus_model(ckpt):
    if not len(ckpt):
        ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt"
    if ckpt.startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=False)
    from s3prl.upstream.wavlm.expert import UpstreamExpert as _UpstreamExpert
    model = _UpstreamExpert(ckpt, model_config=None)
    return model.state_dict()

def save_base_model(model_name, output_dir, model_state_dict, config):
    os.makedirs(output_dir, exist_ok=True)
    output_model_path = os.path.join(output_dir, 'merged_model.pt')
    function_name = f"save_{model_name}_model"
    if callable(eval(function_name)): 
        func = eval(function_name)(output_model_path, model_state_dict)
    else:
        raise NotImplementedError(f"Model hasn't been implemented yet.")
    output_config_path = os.path.join(output_dir, 'config.yaml')
    with open(output_config_path, 'w') as file:
        yaml.safe_dump(config, file, default_flow_style=False)

def save_hubert_model(output_model_path, model_state_dict):
    ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/hubert_base_ls960.pt"
    ckpt = _urls_to_filepaths(ckpt, refresh=False)
    ckpt = torch.load(ckpt)
    model_state_dict = {key.replace('model.', ''): value for key, value in model_state_dict.items()}
    ckpt['model_weight'] = model_state_dict
    torch.save(ckpt, output_model_path)
    print(f"Model saved to {output_model_path}")

def save_wav2vec2_model(output_model_path, model_state_dict):
    ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wav2vec_small.pt"
    ckpt = _urls_to_filepaths(ckpt, refresh=False)
    ckpt = torch.load(ckpt)
    model_state_dict = {key.replace('model.', ''): value for key, value in model_state_dict.items()}
    ckpt['model_weight'] = model_state_dict
    torch.save(ckpt, output_model_path)
    print(f"Model saved to {output_model_path}")

def save_decoar2_model(output_model_path, model_state_dict):
    ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/checkpoint_decoar2.pt"
    ckpt = _urls_to_filepaths(ckpt, refresh=False)
    ckpt = torch.load(ckpt)
    model_state_dict = {key.replace('model.', ''): value for key, value in model_state_dict.items()}
    ckpt['model'] = model_state_dict
    torch.save(ckpt, output_model_path)
    print(f"Model saved to {output_model_path}")

def save_wavlm_base_plus_model(output_model_path, model_state_dict):
    ckpt = "https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt"
    ckpt = _urls_to_filepaths(ckpt, refresh=False)
    ckpt = torch.load(ckpt)
    model_state_dict = {key.replace('model.', ''): value for key, value in model_state_dict.items()}
    ckpt['model'] = model_state_dict
    torch.save(ckpt, output_model_path)
    print(f"Model saved to {output_model_path}") 

def load_finetune_model(finetune_model_paths: List[str]):
    finetune_model_state_dict = []
    for pth in finetune_model_paths:
        ckpt = torch.load(pth, map_location='cpu', weights_only=False)
        if 'Upstream' in ckpt: # For s3prl downstream model
            state_dict = ckpt['Upstream']
        elif 'state_dict' in ckpt: # For SPIN
            state_dict = {k.replace('encoder.model', 'model'): v.clone() for k, v in ckpt['state_dict'].items() if k.startswith('encoder.model')}
            state_dict['model.final_proj.weight'] = None
            state_dict['model.final_proj.bias'] = None
        elif 'model' in ckpt: # For aishell3 continuous pre-training
            state_dict = {'model.'+k: v.clone() for k, v in ckpt['model'].items()}
            state_dict['model.label_embs_concat'] = None
        else:
            print(f"No state dictionary has been identified in the model checkpoint")
            exit(0)
        finetune_model_state_dict.append(state_dict)
    return finetune_model_state_dict
