# 🚀 Speech-FT
This is the official repository of the paper:
[Speech-FT: Merging Pre-trained and Fine-Tuned Speech Representation Models for Cross-Task Generalization](https://ieeexplore.ieee.org/document/11263888). 

- **Speech-FT is a supervised fine-tuning method designed to enhance pre-trained speech encoders**. 
- It effectively addresses the challenge of fine-tuning speech encoders while preserving cross-task generalization ability. 
- For example, when fine-tuning HuBERT on ASR, Speech-FT reduces the phoneme error rate (PER) from 5.17% to 3.94% and improves speaker identification (SID) accuracy from 81.86% to 84.11%. - **Overall, Speech-FT provides a simple yet effective solution for improving pre-trained speech encoders**.

![An overview of Speech-FT](asset/overview.png)

## 🎯 Results of Speech-FT

When fine-tuning HuBERT on ASR with TED-LIUM, Speech-FT reduces the PER from 5.17% to 3.94% and improves the SID accuracy from 81.86% to 84.11%.
![Performance of HuBERT fine-tuning with Speech-FT on SUPERB](asset/performance.png)
Please see [the paper](https://ieeexplore.ieee.org/document/11263888) for full SUPERB evaluation results. 

## 💾 Model Checkpoint

Model checkpoints when fine-tuning with ASR on TED-LIUM3
| Models | URL |
| :--- | :--- |
| HuBERT + Speech-FT | [link](https://drive.google.com/file/d/13yiv5-6SY4dIMarCidJ0FKBPcID1iKG_/view?usp=sharing) |
| wav2vec 2.0 + Speech-FT | [link](https://drive.google.com/file/d/1d8412DKVFeS8vRzE9qL-gGz4IFs2EjQ9/view?usp=sharing) |

## ⚙️ Installation
This codebase is built upon [s3prl](https://github.com/s3prl/s3prl/tree/main).  
Please clone the s3prl repository into `S3PRL_ROOT` and copy the modified files using:
```
cp -r s3prl_modified/* S3PRL_ROOT/s3prl/
```
Then, install s3prl manually (python=3.9 is recommended):
```
cd S3PRL_ROOT
pip install -e ".[all]"
```
## 🔥 Enhance Speech Encoders with Speech-FT
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