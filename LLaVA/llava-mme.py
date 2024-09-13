from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import os
model_path = "../model_weights/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda",
    device="cuda"
)

## Need to modify
output_dir = ""


folders_and_files = [
    {
        "image_folder": "MME_Benchmark_release_version/artwork/images",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/artwork.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/celebrity/images",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/celebrity.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/code_reasoning",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/code_reasoning.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/color",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/color.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/commonsense_reasoning",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/commonsense_reasoning.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/count",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/count.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/existence",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/existence.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/landmark/images",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/landmark.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/numerical_calculation",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/numerical_calculation.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/OCR",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/OCR.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/position",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/position.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/posters/images",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/posters.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/scene/images",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/scene.txt"
    },
    {
        "image_folder": "MME_Benchmark_release_version/text_translation",
        "output_file_path": f"MME_Benchmark_release_version/eval_tool/{output_dir}/text_translation.txt"
    },
]

for item in folders_and_files:
    image_folder = item["image_folder"]
    output_file_path = item["output_file_path"]

    with open(output_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        idx = 0
        for line in lines:
            print(f"Processing {idx} in {output_file_path}")
            idx += 1
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                image_file = parts[0]
                question = parts[1]
                image_path = os.path.join(image_folder, image_file)
                args = type('Args', (), {
                    "query": question,
                    "conv_mode": None,
                    "image_file": image_path,
                    "sep": ",",
                    "temperature": 0,
                    "top_p": None,
                    "num_beams": 1,
                    "max_new_tokens": 512,
                    "device_map":"cuda",
                    "device":"cuda:0",
                    "model": model,
                    "tokenizer": tokenizer,
                    "image_processor":image_processor,
                    "context_len":context_len,
                    "model_name": get_model_name_from_path(model_path),
                    "output_hidden_states": True,
                    "cd_alpha":1,
                    "cd_beta":0.1,
                    "noise_step":500,
                    "use_cd":False,
                    "use_dola":False,
                    "opera_decoding":True
                })()
                output_text = eval_model(args)
                output_text = output_text.replace('\n', ' ')
                output_text = output_text.replace('\t', ' ')
                output_line = f"{line.strip()}\t{output_text}\n"
                output_file.write(output_line)

    print(f"Finished processing {output_file_path}")

print("All files have been processed!")