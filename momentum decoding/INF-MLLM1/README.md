## Introduction

This documentation is about the OPERA decoding in INF-MLLM1, please follow the usege part to run it.

## Usage
1) Please first download official INF-MLLM code from:
   ```
   git clone https://github.com/infly-ai/INF-MLLM.git
   ```
2) To creat the environment about INF-MLLM1,
   ```
   cd INF-MLLM1
   conda create -n infmllm python=3.9
   conda activate infmllm
   conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3) To get the model weights ```InfMLLM-7B-Chat```, please download it from
   ```
   https://huggingface.co/mightyzau/InfMLLM_7B_Chat/tree/main
   ```
So far, we have created a environment about INF-MLLM1, then we will introduce how to apply OPERA.

4) Please replace the ```~/.conda/envs/infmllm/lib/python3.9/site-packages/transformers/generation/utils.py``` of the original environment with ours.
