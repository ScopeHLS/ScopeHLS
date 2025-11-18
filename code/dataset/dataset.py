import os
import json
import math
import torch
import pickle
from torch.utils.data import Dataset
from typing import List
from transformers import AutoTokenizer
from config.config import CONFIG

from .data_util import (
    process_design_params,
    pragma_scope_map,
    code_scope_map,
    char_to_token,
    build_scope_mask
)

# versions: List[str] = ['v18', 'v20', 'v21']
class HLSDesignDataset(Dataset):
    def __init__(self, data_dir: str, versions: List[str] = ['v21'], use_cache: bool = True):
        self.data_dir = data_dir
        self.versions = versions
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG.codet5p_tokenizer_path)

        self.designs = []
        self.src_codes = {}
        self.valid_data = []
        self.invalid_data = []

        self.save_valid_data = []
        self.save_invalid_data = []

        self.max_param_len = 0
        self.max_code_len = 0

        if use_cache and self.load_cached_data():
            return
        
        self._load_data()
        self._preprocess_all()
        
        if use_cache:
            self.save_processed_data()
        
    def _load_data(self):
        """Load all design data from JSON files"""
        print("Loading design data...")
        
        for version in self.versions:
            design_path = os.path.join(self.data_dir, "data", "designs", version)
            
            for fname in os.listdir(design_path):
                if 'json' not in fname:
                    continue
                    
                with open(os.path.join(design_path, fname), 'r') as f:
                    design_points = json.load(f)
                
                kernel_name = fname.split('.')[0]
                if kernel_name == 'stencil':
                    kernel_name = 'stencil_stencil2d'
                
                if kernel_name not in self.src_codes:
                    self.src_codes[kernel_name] = self._load_src_code(kernel_name)
                
                for key, points in design_points.items():
                    design_data = {
                        'kernel_name': kernel_name,
                        'version': version,
                        'design_key': key,
                        'design_params': points['point'],
                        'valid': points['valid'],
                        'perf': points['perf'],
                        'res_util': points['res_util']
                    }
                    self.designs.append(design_data)
        
        print(f"Loaded {len(self.designs)} design points")
        
    def _load_src_code(self, kernel_name: str) -> str:
        """Load source code for a kernel"""
        src_path = os.path.join(self.data_dir, "data", "sources", f"{kernel_name}_kernel.c")
        try:
            with open(src_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Source code not found for {kernel_name}")
            return ""

    def _preprocess_all(self):
        """Preprocess all design points (tokenization, scope mapping, categorization)"""
        kernel_name_list = []
        for d in self.designs:
            kernel_name = d['kernel_name']
            src_code = self.src_codes[kernel_name]
            design_params = d['design_params']

            new_design_params = process_design_params(design_params)

            scope_map_pragma = pragma_scope_map(new_design_params, design_params)
            scope_map_code = code_scope_map(src_code, design_params)

            param_tokens = self.tokenizer(
                new_design_params,
                return_offsets_mapping=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=160,
                add_special_tokens=True
            )
            code_tokens = self.tokenizer(
                src_code,
                return_offsets_mapping=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=1024,
                add_special_tokens=True
            )
            param_len = param_tokens["input_ids"].shape[1]
            code_len = code_tokens["input_ids"].shape[1]

            
            self.max_param_len = max(self.max_param_len, param_len)
            self.max_code_len = max(self.max_code_len, code_len)

            scope_map_pragma_token = char_to_token(new_design_params, param_tokens["offset_mapping"], scope_map_pragma)
            scope_map_code_token = char_to_token(src_code, code_tokens["offset_mapping"], scope_map_code)

            scope_mask = build_scope_mask(design_params, scope_map_pragma_token, scope_map_code_token)
            param_att = param_tokens["attention_mask"][0].unsqueeze(1).float()  # [param_len, 1]
            code_att = code_tokens["attention_mask"][0].unsqueeze(0).float()    # [1, code_len]
            att_mask = torch.matmul(param_att, code_att)  # [param_len, code_len]
            scope_mask = scope_mask * att_mask

            if d['perf'] ==0:
                perf = -5
            else:
                perf = 0.5*math.log2(1e7 / d['perf'])

            item = {
                "param_tokens": param_tokens,
                "code_tokens": code_tokens,
                "scope_mask": scope_mask,
                "design_key": d['design_key'],
                "valid": d['valid'],
                "latency": d['perf'],
                "perf": perf,
                "res_util": d['res_util'],
                "kernel_name": kernel_name,
                "version": d['version']
            }

            save_item = {
                "design_params": design_params,
                "param_tokens": param_tokens,
                "code_tokens": code_tokens,
                "scope_map_pragma_token": scope_map_pragma_token,
                "scope_map_code_token": scope_map_code_token,
                "design_key": d['design_key'],
                "valid": d['valid'],
                "latency": d['perf'],
                "perf": perf,
                "res_util": d['res_util'],
                "kernel_name": kernel_name,
                "version": d['version']
            }

            if d['valid'] is False and d['perf'] == 0:
                self.invalid_data.append(item)
                self.save_invalid_data.append(save_item)
            else:
                self.valid_data.append(item)
                self.save_valid_data.append(save_item)
        print(f"kernels:{kernel_name_list}")
        print(f"Preprocessing done: {len(self.valid_data)} valid, {len(self.invalid_data)} invalid")

    def __len__(self):
        return len(self.valid_data) + len(self.invalid_data)

    def __getitem__(self, idx):
        if idx < len(self.valid_data):
            return self.valid_data[idx]
        else:
            return self.invalid_data[idx - len(self.valid_data)]

    def get_max_lengths(self):
        return self.max_param_len, self.max_code_len
    
    def _get_cache_path(self):
        cache_name = "_".join(self.versions)
        return os.path.join(self.data_dir, f"cached_dataset_{cache_name}.pkl")

    def save_processed_data(self):
        cache_path = self._get_cache_path()
        with open(cache_path, "wb") as f:
            pickle.dump({
                "valid_data": self.save_valid_data,
                "invalid_data": self.save_invalid_data,
            }, f)
        print(f"[✓] Processed dataset saved to {cache_path}")

    def load_cached_data(self):
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            print(f"[!] Cache file {cache_path} not found.")
            return False
        with open(cache_path, "rb") as f:
            print("[*] load cache data...")
            cache = pickle.load(f)
            self.valid_data = cache["valid_data"]
            self.invalid_data = cache["invalid_data"]
            print("[*] Restoring scope masks from saved token maps...")
            self.valid_data, self.invalid_data = self.restore_scope_mask()
        print(f"[✓] Loaded cached dataset from {cache_path}")
        return True

    def restore_scope_mask(self):
        restored_valid = []
        restored_invalid = []

        def restore_list(data_list):
            restored_list = []
            for data in data_list:
                param_tokens = data["param_tokens"]
                code_tokens = data["code_tokens"]
                scope_map_pragma_token = data["scope_map_pragma_token"]
                scope_map_code_token = data["scope_map_code_token"]
                design_params = data["design_params"]

                scope_mask = build_scope_mask(
                    design_params,
                    scope_map_pragma_token,
                    scope_map_code_token
                )
                param_att = param_tokens["attention_mask"][0].unsqueeze(1).float()
                code_att = code_tokens["attention_mask"][0].unsqueeze(0).float()
                att_mask = torch.matmul(param_att, code_att)
                scope_mask = scope_mask * att_mask

                new_data = data.copy()
                new_data["scope_mask"] = scope_mask

                del new_data["scope_map_pragma_token"]
                del new_data["scope_map_code_token"]
                del new_data["design_params"]

                restored_list.append(new_data)
            return restored_list

        restored_valid = restore_list(self.valid_data)
        restored_invalid = restore_list(self.invalid_data)

        return restored_valid, restored_invalid
    
    def get_max_latency(self):
        max_latency = float('-inf')
        best_item = None
        all_data = self.valid_data
        for item in all_data:
            if item['latency'] > max_latency:
                max_latency = item['latency']
                best_item = item
        return max_latency, best_item
    
    def count_high_latency(self, threshold=1e8):
        count = 0
        for item in self.valid_data:
            if item['latency'] > threshold:
                count += 1
                print(f"version: {item['version']}, kernel_name: {item['kernel_name']}, key: {item['design_key']}")
        print(f"\nThere are {count} design points with latency > {threshold}")
