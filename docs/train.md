# 🔥 Fine-tuning & Merging

Speech-FT consists of two elements: **stable fine-tuning** and **weight-space interpolation**.

### Stable Fine-Tuning
To perform stable fine-tuning of HuBERT on ASR with TED-LIUM, run the following command: 
```
python3 S3PRL_ROOT/s3prl/preprocess/preprocess_ted.py -o S3PRL_ROOT/s3prl/data/ted
bash finetune_s3prl_model.sh python3 hubert asr_ted train asr_ted_hubert S3PRL_ROOT/s3prl/downstream/asr_ted/config_finetune.yaml S3PRL_ROOT
```
Note that you should use an absolute path instead of a relative path.

### Weight-Space Interpolation
Modify the `finetune_model_paths` and `output_dir` entries in the merging config file (e.g., `merge_config/ted_asr/hubert_alpha_0.25.yaml`).  
- `finetune_model_paths`: fine-tuned model path generated from stable fine-tuning  
- `output_dir`: the directory to save the interpolated model

Then run:
```
python3 merge_model.py -c merge_config/ted_asr/hubert_alpha_0.25.yaml
```