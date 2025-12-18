import yaml
import torch
from typing import List
from s3prl.upstream.utils import merge_with_parent
import sys

ckpt_pth = sys.argv[1]
upstream_name = sys.argv[2]
assert upstream_name in ['hubert', 'wav2vec2'], "Upstream should be either HuBERT or wav2vec 2.0."
if upstream_name == 'hubert':
    from s3prl.upstream.hubert.convert import load_converted_model
elif upstream_name == 'wav2vec2':
    from s3prl.upstream.wav2vec2.convert import load_converted_model
else:
    raise NotImplementedError

model, task_cfg = load_converted_model(ckpt_pth)
model.feature_grad_mult = 0.0
model.encoder.layerdrop = 0.0
model.eval()
batch_size = 4
padded_wav_len = 16000
padded_wav = torch.rand(batch_size, padded_wav_len)

if task_cfg.normalize:
    padded_wav = F.layer_norm(padded_wav, padded_wav.shape)
# Padding mask need to be changed according to your situation
wav_padding_mask = torch.zeros_like(padded_wav, dtype=torch.bool)
# Padding mask need to be changed according to your situation
with torch.no_grad():
    res = model.extract_features(
        padded_wav,
        padding_mask=wav_padding_mask,
        mask=None,
    )
if upstream_name == 'hubert':
    features = res[0]
elif upstream_name == 'wav2vec2':
    features = res['x']
else:
    raise NotImplementedError
    
print(features.shape)