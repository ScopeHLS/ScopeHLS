import re
import torch
from collections import defaultdict
from config.config import CONFIG

def process_design_params(params):
    type_map = CONFIG.type_map
    parts = []
    for key, value in params.items():
        trimmed = key.strip('_')
        split_parts = trimmed.split("__")
        if len(split_parts) != 2:
            name = key
        else:
            type_code, level = split_parts
            type_name = type_map.get(type_code, type_code.lower())
            name = f"{type_name}-{level}"
        parts.append(f"{name}={value}")
    return ", ".join(parts)

def pragma_scope_map(new_design_params, design_params):
    scope_map = defaultdict(list)
    text_parts = new_design_params.split(", ")
    
    if len(text_parts) != len(design_params):
        raise ValueError("The number of string segments does not match the number of key-value pairs!")

    for key, value in design_params.items():
        text_part = text_parts.pop(0)
        start = new_design_params.find(text_part)
        end = start + len(text_part)
        scope_map[key].append((start, end))
    
    return scope_map

def code_scope_map(src_code: str, design_params: dict):
    scope_map = defaultdict(list)

    pragma_pattern = re.compile(r'#pragma\s+ACCEL\s+(?:PIPELINE|PARALLEL|TILE)[^\n]*\{(__[A-Z]+__L\d+)\}')
    pragma_matches = [(m.start(), m.group(1)) for m in pragma_pattern.finditer(src_code)]

    for_pattern = re.compile(r'for\s*\([^)]*\)\s*\{')
    for_matches = [(m.start(), m.end()) for m in for_pattern.finditer(src_code)]

    def find_loop_body(start_idx):
        stack = []
        i = start_idx
        while i < len(src_code):
            if src_code[i] == '{':
                stack.append(i)
            elif src_code[i] == '}':
                stack.pop()
                if not stack:
                    return start_idx, i + 1
            i += 1
        return start_idx, i

    loop_ranges = []
    for start, end in for_matches:
        body_start = src_code.find('{', end - 1)
        if body_start != -1:
            loop_start, loop_end = find_loop_body(body_start)
            loop_ranges.append((start, loop_start, loop_end))

    for pragma_pos, param in pragma_matches:
        for start, body_start, body_end in loop_ranges:
            if start > pragma_pos:
                scope_map[param].append((start, body_end))
                break

    return scope_map

def char_to_token(design_params, offsets, scope_map):
    scope_map_token = defaultdict(list)
    for param in design_params:
        ranges = scope_map.get(param, [])
        for start, end in ranges:
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start <= start < token_end:
                    start_token_index = i
                if token_start < end <= token_end:
                    end_token_index = i
            scope_map_token[param].append((start_token_index, end_token_index))
    return scope_map_token

def build_scope_mask(design_params, scope_map_pragma_token, scope_map_code_token):
    scope_mask = torch.full((160, 1024), 0.05)

    for param in design_params:
        ranges_pragma = scope_map_pragma_token[param]
        ranges_code = scope_map_code_token[param]
        for (start_p, end_p) in ranges_pragma:
            for (start_c, end_c) in ranges_code:
                scope_mask[start_p:end_p+1, start_c:end_c+1] = 1
    return scope_mask

def build_scope_mask_inference(design_params, scope_map_pragma_token, scope_map_code_token, code_len, pragma_len):
    scope_mask = torch.full((pragma_len, code_len), 0.05)

    for param in design_params:
        ranges_pragma = scope_map_pragma_token[param]
        ranges_code = scope_map_code_token[param]
        for (start_p, end_p) in ranges_pragma:
            for (start_c, end_c) in ranges_code:
                scope_mask[start_p:end_p+1, start_c:end_c+1] = 1
    return scope_mask