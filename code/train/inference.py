import os
import torch
from transformers import AutoTokenizer

from dataset.data_util import char_to_token, code_scope_map, pragma_scope_map, process_design_params, build_scope_mask_inference

from config.config import CONFIG
demo_data_DIR = CONFIG.demo_data_dir

def inference_data():
    # kernel_name = 'bicg-large'
    
    # design_params = {
    #     "__PARA__L0": 5,
    #     "__PARA__L1": 1,
    #     "__PIPE__L0": "",
    #     "__TILE__L0": 1}
    
    kernel_name = '2mm'
    design_params = {
            "__PARA__L0": 1,
            "__PARA__L1": 2,
            "__PARA__L2": 2,
            "__PARA__L3": 1,
            "__PARA__L4": 32,
            "__PARA__L5": 1,
            "__PIPE__L0": "",
            "__PIPE__L1": "off",
            "__PIPE__L2": "",
            "__PIPE__L3": "flatten",
            "__TILE__L0": 1,
            "__TILE__L1": 1,
            "__TILE__L2": 1,
            "__TILE__L3": 80
        }
    
    src_path = os.path.join(demo_data_DIR, "data", "sources", f"{kernel_name}_kernel.c")
    with open(src_path, "r") as f:
        src_code = f.read()
    
    new_design_params = process_design_params(design_params)
    scope_map_pragma = pragma_scope_map(new_design_params, design_params)
    scope_map_code = code_scope_map(src_code, design_params)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG.codet5p_tokenizer_path)
    code_tokens = tokenizer(
        new_design_params,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
        add_special_tokens=True)
    
    param_tokens = tokenizer(
        new_design_params,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        padding=False,
        add_special_tokens=True)
    
    code_len = code_tokens['input_ids'].shape[1]
    pragma_len = param_tokens['input_ids'].shape[1]

    scope_map_pragma_token = char_to_token(new_design_params, param_tokens["offset_mapping"], scope_map_pragma)
    scope_map_code_token = char_to_token(src_code, code_tokens["offset_mapping"], scope_map_code)

    scope_mask = build_scope_mask_inference(design_params, scope_map_pragma_token, scope_map_code_token, code_len, pragma_len)
    param_att = param_tokens["attention_mask"][0].unsqueeze(1).float()  # [param_len, 1]
    code_att = code_tokens["attention_mask"][0].unsqueeze(0).float()    # [1, code_len]
    att_mask = torch.matmul(param_att, code_att)  # [param_len, code_len]
    scope_mask = scope_mask * att_mask

    return code_tokens, new_design_params, tokenizer, scope_mask
