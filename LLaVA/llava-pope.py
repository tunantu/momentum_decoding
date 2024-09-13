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



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="momentum")
parser.add_argument("--setting", type=str, default="popular")
parser.add_argument("--use_dola", action='store_true', default=False)
parser.add_argument("--use_cd", action='store_true', default=False)
parser.add_argument("--use_opera", action='store_true', default=False)
args1 = parser.parse_args()



import json
input_file_path = f'./POPE/aokvqa/{args1.setting}/aokvqa_pope_seem_{args1.setting}.json'
output_file_path = f'./POPE/aokvqa/{args1.setting}/{args1.output_dir}.json'

print(args1)


new_data = []



### For POPE coco
# with open(input_file_path, 'r') as file:
#     for line in file:
#         data = json.loads(line.strip())
#         image = data.get('image')
#         question = data.get('text')
#         question_id = data.get('question_id')
#         image_path = f"vqadataset/val2014/{image}"
#         args = type('Args', (), {
#             "query": question,
#             "conv_mode": None,
#             "image_file": image_path,
#             "sep": ",",
#             "temperature": 0,
#             "top_p": None,
#             "num_beams": 1,
#             "max_new_tokens": 512,
#             "device_map":"cuda",
#             "device":"cuda",
#             "model": model,
#             "tokenizer": tokenizer,
#             "image_processor":image_processor,
#             "context_len":context_len,
#             "model_name": get_model_name_from_path(model_path),
#             "output_hidden_states": True,
#             "cd_alpha":1,
#             "cd_beta":0.1,
#             "noise_step":500,
#             "use_cd":args1.use_cd,
#             "use_dola":args1.use_dola,
#         })()
#         output_text = eval_model(args)
#         output_text = output_text.replace('\n', ' ')
#         output_text = output_text.replace('\t', ' ')
#         new_entry = {
#             'question': question,
#             'answer': output_text
#         }
        
#         new_data.append(new_entry)
#         print(f"Processing {question_id} ing")



## For POPE aokvqa
with open(input_file_path, 'r') as file:
    datas = json.load(file)
for data in datas:
    image = data.get('image')
    question = data.get('text')
    question_id = data.get('question_id')
    image_path = f"vqadataset/val2014/{image}"
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
        "device":"cuda",
        "model": model,
        "tokenizer": tokenizer,
        "image_processor":image_processor,
        "context_len":context_len,
        "model_name": get_model_name_from_path(model_path),
        "output_hidden_states": True,
        "cd_alpha":1,
        "cd_beta":0.1,
        "noise_step":500,
        "use_cd":args1.use_cd,
        "use_dola":args1.use_dola,
        "use_opera":args1.use_opera
    })()
    output_text = eval_model(args)
    output_text = output_text.replace('\n', ' ')
    output_text = output_text.replace('\t', ' ')
    new_entry = {
        'question': question,
        'answer': output_text
    }
    
    new_data.append(new_entry)
    print(f"Processing {question_id} ing")


with open(output_file_path, 'w') as outfile:
    for entry in new_data:
        json.dump(entry, outfile)
        outfile.write('\n')

print(f"Finished: {output_file_path}")


# CUDA_VISIBLE_DEVICES=0