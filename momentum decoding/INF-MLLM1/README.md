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

5) Then, please replace the model weight file ```InfMLLM_7B_Chat/modeling_infmllm_chat.py``` with our updated version.

6) Please put the folder vcd_utils in ```./INF-MLLM1```.

7) For dataset preparation, please put the given folder ```POPE``` and ```eval_tool``` in ```./ INF-MLLM1```, where   ```eval_tool``` is for MME. Meanwhile, the dataset ```MME_Benchmark_release_version``` should also be put in ```./INF-MLLM1```.

9) Then, for MME dataset, please run:
   ```
   CUDA_VISIBLE_DEVICES=0 python inf-mme.py --model_path "./InfMLLM_7B_Chat" --use_opera
   ```

10) For POPE (coco), please run:
    ```
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset coco --setting random --output_dir opera  --use_opera
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset coco --setting popular --output_dir opera  --use_opera
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset coco --setting adversarial --output_dir opera  --use_opera 
    ```
11) For POPE(aokvqa), please run:
    ```
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset aokvqa --setting random --output_dir opera  --use_opera
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset aokvqa --setting popular --output_dir opera  --use_opera
    CUDA_VISIBLE_DEVICES=0 python inf-pope.py  --dataset aokvqa --setting adversarial --output_dir opera  --use_opera

    ```
