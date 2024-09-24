from vcd_utils.vcd_sample import evolve_vcd_sampling
evolve_vcd_sampling()
import os, sys 
rootdir = os.path.abspath(os.path.dirname(__file__))
if rootdir not in sys.path:
    sys.path.insert(0, rootdir)

import re
import torch
from PIL import Image 
import requests
from transformers import AutoModel, AutoTokenizer

from evaluate.infmllm_chat.utils import tokenizer_image_token
from evaluate.infmllm_chat.conversation import conv_templates, SeparatorStyle

IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def expand2square(pil_img, background_color):
    # pad to middle for square shape
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_prompt(conv_mode, question, history=[]):
    conv = conv_templates[conv_mode].copy()
    if len(history) == 0:
        question = DEFAULT_IMAGE_TOKEN + '\n' + question
    else:
        if DEFAULT_IMAGE_TOKEN not in history[0][0]:
            history[0][0] = DEFAULT_IMAGE_TOKEN + '\n' + history[0][0]

    for qa in history:
        conv.append_message(conv.roles[0], qa[0])
        conv.append_message(conv.roles[1], qa[1])

    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt = conv.get_prompt()
    return prompt

def generate(model, tokenizer, stop_str, input_ids, image_tensor):
    if args.use_cd:
        from vcd_utils.vcd_add_noise import add_diffusion_noise
        image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).cuda().to(dtype=torch.bfloat16),
                images_cd=(image_tensor_cd.unsqueeze(0).cuda().to(dtype=torch.bfloat16) if image_tensor_cd is not None else None),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                output_hidden_states=args.output_hidden_states)
            
    elif args.use_dola:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_hidden_states=args.output_hidden_states,
                dola_decoding=True,
                mature_layer=32,
                base_layer=None,
                candidate_premature_layers=[0,2,4,6,8,10,12,14],
                relative_top= 0,
                contrastive_decoding=None,
                student_model = None,
                )
            
    elif args.use_opera:
        key_position = {
            "image_start": 35,
            "image_end": 1378,
            "response_start": input_ids.size(1) + 1344-1,
            }
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=args.output_hidden_states,
                num_beams=2,
                output_attentions=True,
                opera_decoding=True,
                scale_factor=50,
                threshold=25,
                num_attn_candidates=1,
                penalty_weights=1,
                key_position=key_position,
                )
                        
            
    else:
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                output_hidden_states=args.output_hidden_states)
        
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs
        
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./InfMLLM_7B_Chat")
parser.add_argument("--conv_mode", type=str, default="vicuna_v1")
parser.add_argument("--temperature", type=float, default=0)
parser.add_argument("--top_p", type=float, default=None)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--output_hidden_states", type=bool, default=True)
parser.add_argument("--noise_step", type=int, default=500)
parser.add_argument("--cd_alpha", type=float, default=1)
parser.add_argument("--cd_beta", type=float, default=0.1)
parser.add_argument("--output_dir", type=str, default="baseline")
parser.add_argument("--use_cd", action='store_true', default=False)
parser.add_argument("--use_dola", action='store_true', default=False)
parser.add_argument("--use_opera", action='store_true', default=False)
parser.add_argument("--dataset", type=str, default="coco")
parser.add_argument("--setting", type=str, default="popular")
args = parser.parse_args()



disable_torch_init()
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16) 
# model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, load_in_8bit=True)
model = model.cuda().eval()
image_processor = model.get_model().get_vision_tower().image_processor

stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2   # </s>

print(args)


if args.dataset == "coco":
    import json
    input_file_path = f'./POPE/coco/{args.setting}/coco_pope_{args.setting}.json'
    output_file_path = f'./POPE/coco/{args.setting}/{args.output_dir}.json'
    new_data = []
    with open(input_file_path, 'r') as file:
        for line in file:
            data = json.loads(line.strip())
            image = data.get('image')
            question = data.get('text')
            question_id = data.get('question_id')
            image_path = f"/val2014/{image}"

            prompt = get_prompt(args.conv_mode, question)
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
            input_ids = input_ids.to(device='cuda', non_blocking=True)

            raw_image = Image.open(image_path).convert('RGB')
            image = expand2square(raw_image, tuple(int(x*255) for x in image_processor.image_mean))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            output_text = generate(model, tokenizer, stop_str, input_ids, image_tensor)
            output_text = output_text.replace('\n', ' ')
            output_text = output_text.replace('\t', ' ')
            new_entry = {
                'question': question,
                'answer': output_text
            }
            
            new_data.append(new_entry)
            print(f"Processing {question_id} ing")


elif args.dataset =="aokvqa":
    import json
    input_file_path = f'./POPE/aokvqa/{args.setting}/aokvqa_pope_seem_{args.setting}.json'
    output_file_path = f'./POPE/aokvqa/{args.setting}/{args.output_dir}.json'
    new_data = []
    with open(input_file_path, 'r') as file:
        datas = json.load(file)
    for data in datas:
        image = data.get('image')
        question = data.get('text')
        question_id = data.get('question_id')
        image_path = f"/home/shibin/kaishen/editing/manual_data/vqadataset/val2014/{image}"

        prompt = get_prompt(args.conv_mode, question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # print(input_ids)

        raw_image = Image.open(image_path).convert('RGB')
        image = expand2square(raw_image, tuple(int(x*255) for x in image_processor.image_mean))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        # print(image_tensor.shape)   3, 448, 448

        output_text = generate(model, tokenizer, stop_str, input_ids, image_tensor)
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
