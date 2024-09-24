#### This part is for baseline and vcd



# Thanks to the open source code of LLaVA-1.5

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .eva_vit import EVACLIPVisionTower
from .pooler import Pooler

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200


class LlavaMetaModel:
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.vision_tower = EVACLIPVisionTower(config.image_size)
        self.mm_projector = Pooler(config.mm_hidden_size, config.hidden_size, 
                                   pool_out_size=config.pool_out_size)

    def get_vision_tower(self):
        return self.vision_tower

class InfMLLMLlamaModel(LlavaMetaModel, LlamaModel):
    def __init__(self, config):
        super(InfMLLMLlamaModel, self).__init__(config)


class InfMLLMMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids,          # [b, L]
        attention_mask,     # [b, L]
        past_key_values,    # None
        labels,             # [b, L]
        images              # [b, 3, 336, 336]
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            # print("test1")
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            # print("test2") #True
            image_features = self.encode_images(images)                             # [b, 576, 5120]

        # print(image_features.shape)  ## 1, 1344, 4096
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while image_token_indices.numel() > 0:
                cur_image_features = image_features[cur_image_idx]
                image_token_start = image_token_indices[0]
                #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                #    cur_new_input_embeds.append(cur_image_features)
                #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                #    if labels is not None:
                #        cur_new_labels.append(cur_labels[:image_token_start])
                #        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                #        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                #        cur_labels = cur_labels[image_token_start+2:]
                #else:

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                cur_new_input_embeds.append(cur_image_features)
                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start])
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                    cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                
                #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                #    cur_input_ids = cur_input_ids[image_token_start+2:]
                #else:
                cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                
            if cur_input_ids.numel() > 0:
                #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                #else:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


class InfMLLMLlamaForCausalLM(LlamaForCausalLM, InfMLLMMetaForCausalLM):

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = InfMLLMLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,

        #################For VCD#######################
        images_cd: Optional[torch.FloatTensor] = None,
        cd_beta: Optional[torch.FloatTensor] = None,
        cd_alpha: Optional[torch.FloatTensor] = None,
        ###############################################
        #######For DOLA################################
        dola_decoding: Optional[bool] = None,
        mature_layer: Optional[int] = None,
        base_layer: Optional[int] = None,
        candidate_premature_layers: Optional[List[int]] = None,
        relative_top: Optional[float] = 0.1,
        contrastive_decoding: Optional[bool] = None,
        student_model = None,
        early_exit_layers=None,
        ##############################################

    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions       # False
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )   # False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict                           # True

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        
        ###############################DOLA################################################
        if early_exit_layers is not None:
            logits_dict = {}
            for i, early_exit_layer in enumerate(early_exit_layers):
                logits = self.lm_head(outputs.hidden_states[early_exit_layer])
                logits_dict[early_exit_layer] = logits
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                
            final_outputs = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            return logits_dict, final_outputs
        #######################################################################
        else:
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_cd(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images_cd", None),
            }
        )
        return model_inputs
    
























# # Thanks to the open source code of LLaVA-1.5

# from abc import ABC, abstractmethod
# from typing import List, Optional, Tuple, Union
# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss

# # from transformers import LlamaModel, LlamaForCausalLM
# from transformers import LlamaModel as LlamaModelOrig
# from transformers import LlamaForCausalLM as LlamaForCausalLMOrig
# from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings
# from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
# from transformers.modeling_outputs import BaseModelOutputWithPast

# from transformers.modeling_outputs import CausalLMOutputWithPast

# from .eva_vit import EVACLIPVisionTower
# from .pooler import Pooler
# from transformers.utils import logging
# logger = logging.get_logger(__name__)

# IGNORE_INDEX = -100
# IMAGE_TOKEN_INDEX = -200



# class LlamaModel(LlamaModelOrig):
#     def custom_cosine_similarity(self, vec1, vec2, eps=1e-8):
#         norm_vec1 = torch.sqrt(torch.sum(vec1 ** 2, dim=-1))
#         norm_vec2 = torch.sqrt(torch.sum(vec2 ** 2, dim=-1))
#         norm_vec1 = torch.clamp(norm_vec1, min=eps)
#         norm_vec2 = torch.clamp(norm_vec2, min=eps)
#         dot_product = torch.sum(vec1 * vec2, dim=-1)
#         cosine_sim = dot_product / (norm_vec1 * norm_vec2)
#         # cosine_sim = torch.clamp(cosine_sim, min=-1.0, max=1.0)
#         if torch.isinf(cosine_sim).any() or torch.isnan(cosine_sim).any():
#             cosine_sim = torch.where(torch.isinf(cosine_sim) | torch.isnan(cosine_sim), torch.zeros_like(cosine_sim), cosine_sim)
        
#         return cosine_sim
#     def compute_vector_difference_magnitude(self, vec1, vec2, eps=1e-8):
#         difference = vec1 - vec2
#         magnitude = torch.sqrt(torch.sum(difference ** 2, dim=-1, keepdim=True))
#         magnitude = torch.clamp(magnitude, min=eps)
#         return magnitude

#     @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         lm_head=None
#     ) -> Union[Tuple, BaseModelOutputWithPast]:
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache

#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # retrieve input_ids and inputs_embeds
#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

#         seq_length_with_past = seq_length
#         past_key_values_length = 0

#         if past_key_values is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length

#         if position_ids is None:
#             device = input_ids.device if input_ids is not None else inputs_embeds.device
#             position_ids = torch.arange(
#                 past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
#             )
#             position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
#         else:
#             position_ids = position_ids.view(-1, seq_length).long()

#         if inputs_embeds is None:
#             inputs_embeds = self.embed_tokens(input_ids)
#         # embed positions
#         if attention_mask is None:
#             attention_mask = torch.ones(
#                 (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
#             )
#         attention_mask = self._prepare_decoder_attention_mask(
#             attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
#         )

#         hidden_states = inputs_embeds

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         all_hidden_states = () if output_hidden_states else None
#         all_self_attns = () if output_attentions else None
#         next_decoder_cache = () if use_cache else None

#         prev_states = 0
#         momentum_flag = False
#         momentum_count=0
#         delta_hidden_states=0
#         prev_hidden_states=0
#         momentum_decoding_flag=False
#         for idx, decoder_layer in enumerate(self.layers):
#             if output_hidden_states:
#                 all_hidden_states += (hidden_states,)

#             past_key_value = past_key_values[idx] if past_key_values is not None else None

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, output_attentions, None)

#                     return custom_forward

#                 layer_outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(decoder_layer),
#                     hidden_states,
#                     attention_mask,
#                     position_ids,
#                     None,
#                 )
#             else:
#                 layer_outputs = decoder_layer(
#                     hidden_states,
#                     attention_mask=attention_mask,
#                     position_ids=position_ids,
#                     past_key_value=past_key_value,
#                     output_attentions=output_attentions,
#                     use_cache=use_cache,
#                 )

            

#             hidden_states = layer_outputs[0]

#             current_incre = hidden_states[:,-1:,:]-all_hidden_states[-1][:,-1:,:]

            
#             ## baseline, 1492.71, 266.07

#             # if idx>=16:   ## demo1    1510, 265
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.3:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.3
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 

#             # if idx>=16:   ## demo2    1506, 271
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.3:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.4
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 

 

#             # if idx>=16:   ## demo4    1513, 273
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.4:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.4
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 
 
#             # if idx>=16:   ## demo5   1518, 273
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.5:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.4
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 


            
#             # if idx>=16:   ## demo6   1520, 269
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.6:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.4
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 



#             # if idx>=16:   ## demo7    1480, 264
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.2
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.5:
#             #         momentum_decoding_flag=True
#             #         prev_hidden_states_momentum = 0.5
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 



#             if idx>=16:   ## POPE
#                 delta_momentum = 0.7
#                 prev_hidden_states_momentum = 0.05
#                 logits_3 = lm_head(hidden_states[:,-1:,:])
#                 logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#                 logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#                 P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#                 P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#                 P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#                 delta_p2 = P_3-P_2
#                 delta_p1 = P_2-P_1
#                 delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#                 cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#                 if cosine_similarity<-0.6:
#                     momentum_decoding_flag=True
#                     prev_hidden_states_momentum = 0.1
#                 prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#                 if momentum_decoding_flag:
#                     hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 



#             # if idx>=16:   ## ablation 
#             #     delta_momentum = 0.7
#             #     prev_hidden_states_momentum = 0.4
#             #     logits_3 = lm_head(hidden_states[:,-1:,:])
#             #     logits_2 = lm_head(all_hidden_states[-1][:,-1:,:])
#             #     logits_1 = lm_head(all_hidden_states[-2][:,-1:,:])
#             #     P_3 = torch.nn.functional.softmax(logits_3, dim=-1)
#             #     P_2 = torch.nn.functional.softmax(logits_2, dim=-1)
#             #     P_1 = torch.nn.functional.softmax(logits_1, dim=-1)
#             #     delta_p2 = P_3-P_2
#             #     delta_p1 = P_2-P_1
#             #     delta_hidden_states = delta_momentum * delta_hidden_states + (1-delta_momentum) * delta_p1
#             #     cosine_similarity = self.custom_cosine_similarity(delta_p2, delta_hidden_states)

#             #     if cosine_similarity<-0.6:
#             #         momentum_decoding_flag=True
#             #         # prev_hidden_states_momentum = 0.4
#             #     prev_hidden_states = prev_hidden_states_momentum * prev_hidden_states + (1-prev_hidden_states_momentum) * current_incre
#             #     if momentum_decoding_flag:
#             #         hidden_states[:, -1:, :] = hidden_states[:, -1:, :] - current_incre + prev_hidden_states 









#             if use_cache:
#                 next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

#             if output_attentions:
#                 all_self_attns += (layer_outputs[1],)

        

#         hidden_states = self.norm(hidden_states)


#         # add hidden states from the last decoder layer
#         if output_hidden_states:
#             all_hidden_states += (hidden_states,)

#         next_cache = next_decoder_cache if use_cache else None
#         if not return_dict:
#             return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
#         return BaseModelOutputWithPast(
#             last_hidden_state=hidden_states,
#             past_key_values=next_cache,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attns,
#         )


# class LlamaForCausalLM(LlamaForCausalLMOrig):
#     # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
#     # @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
        
#         # print("$$$")
#         r"""
#         Args:
#             labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
#                 Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
#                 config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
#                 (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

#         Returns:

#         Example:

#         ```python
#         >>> from transformers import AutoTokenizer, LlamaForCausalLM

#         >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
#         >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

#         >>> prompt = "Hey, are you consciours? Can you talk to me?"
#         >>> inputs = tokenizer(prompt, return_tensors="pt")

#         >>> # Generate
#         >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
#         >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#         "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
#         ```"""

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             position_ids=position_ids,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             lm_head = self.lm_head
#         )

#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)


#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )




# class LlavaMetaModel:
#     def __init__(self, config):
#         super(LlavaMetaModel, self).__init__(config)
#         self.vision_tower = EVACLIPVisionTower(config.image_size)
#         self.mm_projector = Pooler(config.mm_hidden_size, config.hidden_size, 
#                                    pool_out_size=config.pool_out_size)

#     def get_vision_tower(self):
#         return self.vision_tower

# class InfMLLMLlamaModel(LlavaMetaModel, LlamaModel):
#     def __init__(self, config):
#         super(InfMLLMLlamaModel, self).__init__(config)


# class InfMLLMMetaForCausalLM(ABC):

#     @abstractmethod
#     def get_model(self):
#         pass

#     def get_vision_tower(self):
#         return self.get_model().get_vision_tower()

#     def encode_images(self, images):
#         image_features = self.get_model().get_vision_tower()(images)
#         image_features = self.get_model().mm_projector(image_features)
#         return image_features

#     def prepare_inputs_labels_for_multimodal(
#         self, 
#         input_ids,          # [b, L]
#         attention_mask,     # [b, L]
#         past_key_values,    # None
#         labels,             # [b, L]
#         images              # [b, 3, 336, 336]
#     ):
#         vision_tower = self.get_vision_tower()
#         if vision_tower is None or images is None or input_ids.shape[1] == 1:
#             if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
#                 attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
#             return input_ids, attention_mask, past_key_values, None, labels

#         if type(images) is list or images.ndim == 5:
#             concat_images = torch.cat([image for image in images], dim=0)
#             image_features = self.encode_images(concat_images)
#             split_sizes = [image.shape[0] for image in images]
#             image_features = torch.split(image_features, split_sizes, dim=0)
#             image_features = [x.flatten(0, 1) for x in image_features]
#         else:
#             image_features = self.encode_images(images)                             # [b, 576, 5120]

#         new_input_embeds = []
#         new_labels = [] if labels is not None else None
#         cur_image_idx = 0
#         for batch_idx, cur_input_ids in enumerate(input_ids):
#             if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
#                 # multimodal LLM, but the current sample is not multimodal
#                 # FIXME: this is a hacky fix, for deepspeed zero3 to work
#                 half_len = cur_input_ids.shape[0] // 2
#                 cur_image_features = image_features[cur_image_idx]
#                 cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
#                 cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
#                 cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
#                 new_input_embeds.append(cur_input_embeds)
#                 if labels is not None:
#                     new_labels.append(labels[batch_idx])
#                 cur_image_idx += 1
#                 continue

#             image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
#             cur_new_input_embeds = []
#             if labels is not None:
#                 cur_labels = labels[batch_idx]
#                 cur_new_labels = []
#                 assert cur_labels.shape == cur_input_ids.shape

#             while image_token_indices.numel() > 0:
#                 cur_image_features = image_features[cur_image_idx]
#                 image_token_start = image_token_indices[0]
#                 #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
#                 #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
#                 #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
#                 #    cur_new_input_embeds.append(cur_image_features)
#                 #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
#                 #    if labels is not None:
#                 #        cur_new_labels.append(cur_labels[:image_token_start])
#                 #        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
#                 #        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
#                 #        cur_labels = cur_labels[image_token_start+2:]
#                 #else:

#                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
#                 cur_new_input_embeds.append(cur_image_features)
#                 if labels is not None:
#                     cur_new_labels.append(cur_labels[:image_token_start])
#                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
#                     cur_labels = cur_labels[image_token_start+1:]
#                 cur_image_idx += 1
                
#                 #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
#                 #    cur_input_ids = cur_input_ids[image_token_start+2:]
#                 #else:
#                 cur_input_ids = cur_input_ids[image_token_start+1:]
#                 image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
                
#             if cur_input_ids.numel() > 0:
#                 #if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
#                 #    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
#                 #else:
#                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
#                 if labels is not None:
#                     cur_new_labels.append(cur_labels)
#             cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
#             cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
#             new_input_embeds.append(cur_new_input_embeds)
#             if labels is not None:
#                 cur_new_labels = torch.cat(cur_new_labels, dim=0)
#                 new_labels.append(cur_new_labels)

#         if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
#             max_len = max(x.shape[0] for x in new_input_embeds)

#             new_input_embeds_align = []
#             for cur_new_embed in new_input_embeds:
#                 cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
#                 new_input_embeds_align.append(cur_new_embed)
#             new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

#             if labels is not None:
#                 new_labels_align = []
#                 _new_labels = new_labels
#                 for cur_new_label in new_labels:
#                     cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
#                     new_labels_align.append(cur_new_label)
#                 new_labels = torch.stack(new_labels_align, dim=0)

#             if attention_mask is not None:
#                 new_attention_mask = []
#                 for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
#                     new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
#                     new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
#                     cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
#                     new_attention_mask.append(cur_new_attention_mask)
#                 attention_mask = torch.stack(new_attention_mask, dim=0)
#                 assert attention_mask.shape == new_labels.shape
#         else:
#             new_input_embeds = torch.stack(new_input_embeds, dim=0)
#             if labels is not None:
#                 new_labels  = torch.stack(new_labels, dim=0)

#             if attention_mask is not None:
#                 new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
#                 attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
#                 assert attention_mask.shape == new_input_embeds.shape[:2]

#         return None, attention_mask, past_key_values, new_input_embeds, new_labels


# class InfMLLMLlamaForCausalLM(LlamaForCausalLM, InfMLLMMetaForCausalLM):

#     def __init__(self, config):
#         super(LlamaForCausalLM, self).__init__(config)
#         self.model = InfMLLMLlamaModel(config)
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#     def forward(
#         self,
#         input_ids: torch.LongTensor = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         images: Optional[torch.FloatTensor] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple, CausalLMOutputWithPast]:
        
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions       # False
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )   # False
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict                           # True

#         input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

#         # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
#         outputs = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             inputs_embeds=inputs_embeds,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             lm_head=self.lm_head
#         )

#         hidden_states = outputs[0]
#         logits = self.lm_head(hidden_states)

#         loss = None
#         if labels is not None:
#             # Shift so that tokens < n predict n
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             # Flatten the tokens
#             loss_fct = CrossEntropyLoss()
#             shift_logits = shift_logits.view(-1, self.config.vocab_size)
#             shift_labels = shift_labels.view(-1)
#             # Enable model/pipeline parallelism
#             shift_labels = shift_labels.to(shift_logits.device)
#             loss = loss_fct(shift_logits, shift_labels)

#         if not return_dict:
#             output = (logits,) + outputs[1:]
#             return (loss,) + output if loss is not None else output

#         return CausalLMOutputWithPast(
#             loss=loss,
#             logits=logits,
#             past_key_values=outputs.past_key_values,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )

#     def prepare_inputs_for_generation(
#         self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
#     ):
#         if past_key_values:
#             input_ids = input_ids[:, -1:]

#         # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
#         if inputs_embeds is not None and past_key_values is None:
#             model_inputs = {"inputs_embeds": inputs_embeds}
#         else:
#             model_inputs = {"input_ids": input_ids}

#         model_inputs.update(
#             {
#                 "past_key_values": past_key_values,
#                 "use_cache": kwargs.get("use_cache"),
#                 "attention_mask": attention_mask,
#                 "images": kwargs.get("images", None),
#             }
#         )
#         return model_inputs
    