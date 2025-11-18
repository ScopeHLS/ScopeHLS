import torch
import os
import time
import sys
import torch.optim as optim

from config.config import CONFIG
from dataset.dataset import HLSDesignDataset
from dataset.dataloader import get_dataloaders, get_dataloaders_stage2, get_dataloaders_fine_tune
from model.model import pragma_cross_code as Net
from train.train import train_one_epoch, validate, print_losses
from train.inference import inference_data

data_DIR = CONFIG.data_dir
demo_data_DIR = CONFIG.demo_data_dir

def train_main():
    # stage_dataset = HLSDesignDataset(data_dir=data_DIR, versions=['v18', 'v20'])
    stage_dataset = HLSDesignDataset(data_dir=data_DIR, versions=['v21'])

    train_loader, val_loader, test_loader = get_dataloaders(stage_dataset, batch_size=CONFIG.stage1_batch_size)
    device = CONFIG.device
    model = Net()
    # model.load_state_dict(torch.load(CONFIG.model_pth_path, map_location=device))
    model.to(device)

    optimizer_cls = getattr(optim, CONFIG.stage1_optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=CONFIG.stage1_learning_rate)
    
    best_val_loss = float('inf')
    for epoch in range(CONFIG.stage1_epochs):
        mode = "stage1"
        for param in model.parameters():
            param.requires_grad = True
        model.freeze_by_mode(mode)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, mode)
        val_loss = validate(model, val_loader, device, mode)
        if val_loss['valid'] < best_val_loss:
            best_val_loss = val_loss['valid']
            model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage1_path)
            torch.save(model.state_dict(), model_path)
        print_losses(epoch, train_loss, prefix="train")
        print_losses(epoch, val_loss, prefix="val--")

        if epoch % 5 ==0:
            test_loss = validate(model, test_loader, device, mode)
            print_losses(epoch, test_loss, prefix="test")

    train_loader_stage2, val_loader_stage2, test_loader_stage2 = get_dataloaders_stage2(stage_dataset, batch_size=CONFIG.stage2_batch_size)
    device = CONFIG.device
    model = Net()
    model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage1_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    optimizer_cls = getattr(optim, CONFIG.stage2_optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=CONFIG.stage2_learning_rate)
    
    best_val_loss = float('inf')
    for epoch in range(CONFIG.stage2_epochs):
        mode = "stage2"
        for param in model.parameters():
            param.requires_grad = True
        model.freeze_by_mode(mode)
        train_loss = train_one_epoch(model, train_loader_stage2, optimizer, device, mode)
        val_loss = validate(model, val_loader_stage2, device, mode)
        if val_loss['perf'] < best_val_loss:
            best_val_loss = val_loss['perf']
            model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage2_path)
            torch.save(model.state_dict(), model_path)
        print_losses(epoch, train_loss, prefix="train")
        print_losses(epoch, val_loss, prefix="val--")

        if epoch % 5 ==0:
            test_loss = validate(model, test_loader_stage2, device, mode)
            print_losses(epoch, test_loss, prefix="test")
            
    train_loader_stage3, val_loader_stage3, test_loader_stage3 = get_dataloaders(stage_dataset, batch_size=CONFIG.stage3_batch_size)
    device = CONFIG.device
    model = Net()
    model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage2_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    optimizer_cls = getattr(optim, CONFIG.stage3_optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=CONFIG.stage3_learning_rate)

    best_val_loss = float('inf')
    for epoch in range(CONFIG.stage3_epochs):
        mode = "stage3"
        for param in model.parameters():
            param.requires_grad = True
        train_loss = train_one_epoch(model, train_loader_stage3, optimizer, device, mode)
        val_loss = validate(model, val_loader_stage3, device, mode)
        if val_loss['perf'] < best_val_loss:
            best_val_loss = val_loss['perf']
            model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage3_path)
            torch.save(model.state_dict(), model_path)
        print_losses(epoch, train_loss, prefix="train")
        print_losses(epoch, val_loss, prefix="val--")

        if epoch % 5 ==0:
            test_loss = validate(model, test_loader_stage3, device, mode)
            print_losses(epoch, test_loss, prefix="test")


    finetune_dataset = HLSDesignDataset(data_dir=data_DIR, versions=['v21'])
    train_loader_finetune, _, _ = get_dataloaders_fine_tune(finetune_dataset, batch_size=CONFIG.finetune_batch_size)
    device = CONFIG.device
    model = Net()
    model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage3_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    optimizer_cls = getattr(optim, CONFIG.finetune_optimizer)
    optimizer = optimizer_cls(model.parameters(), lr=CONFIG.finetune_learning_rate)

    best_val_loss = float('inf')
    for epoch in range(CONFIG.finetune_epochs):
        mode = "stage3"
        for param in model.parameters():
            param.requires_grad = True
        train_loss = train_one_epoch(model, train_loader_finetune, optimizer, device, mode)
        if train_loss['total'] < best_val_loss:
            best_val_loss = train_loss['total']
            model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_finetune_path)
            torch.save(model.state_dict(), model_path)
        print_losses(epoch, train_loss, prefix="train")


def inference_time_main():
    device = CONFIG.device
    model = Net()
    model_path = os.path.join(CONFIG.model_base_dir, CONFIG.model_stage3_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    code_tokens, design_params, tokenizer, scope_mask = inference_data()

    with torch.no_grad():
        mode = "inference"
        model.eval()

        # Get cached inputs for code tokens
        cahced_code_inputs = model.get_cached_inputs(
            code_tokens["input_ids"].to(device),
            code_tokens["attention_mask"].to(device)
        )
        param_tokens = tokenizer(
            design_params,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            padding=False,
            add_special_tokens=True)
        outputs = model(
            pragma_input_ids = param_tokens['input_ids'].unsqueeze(0).to(device),        # [seq_len] → [1, seq_len]
            code_input = cahced_code_inputs.unsqueeze(0).to(device),                     # [seq_len, dim] → [1, seq_len, dim]
            code_mask = code_tokens["attention_mask"].unsqueeze(0).to(device),           # [seq_len] → [1, seq_len]
            pragma_mask = param_tokens['attention_mask'].unsqueeze(0).to(device),        # [seq_len] → [1, seq_len]
            scope_mask = scope_mask.unsqueeze(0).to(device),    
            mode=mode
        )

        # Repeat inference for 5 minutes
        batch_size = CONFIG.inference_batch_size
        num_iterations = 0
        start_time = time.time()
        elapsed_time = 0

        # Prepare batched inputs
        batched_pragma_input_ids = param_tokens['input_ids'].repeat(batch_size, 1).to(device)
        batched_code_input = cahced_code_inputs.repeat(batch_size, 1, 1).to(device)
        batched_code_mask = code_tokens["attention_mask"].repeat(batch_size, 1).to(device)
        batched_pragma_mask = param_tokens['attention_mask'].repeat(batch_size, 1).to(device)
        batched_scope_mask = scope_mask.repeat(batch_size, 1, 1).to(device)

        while elapsed_time < CONFIG.inference_time:  # 5 minutes = 300 seconds
            start_iteration = time.time()
            for i in range(batch_size):
                param_tokens = tokenizer(
                    design_params,
                    return_offsets_mapping=True,
                    return_tensors="pt",
                    truncation=True,
                    padding=False,
                    add_special_tokens=True)
            outputs = model(
                pragma_input_ids=batched_pragma_input_ids,
                code_input=batched_code_input,
                code_mask=batched_code_mask,
                pragma_mask=batched_pragma_mask,
                scope_mask=batched_scope_mask,
                mode=mode
            )
            end_iteration = time.time()
            elapsed_time = time.time() - start_time
            num_iterations += 1
            sys.stdout.write(f"\rIteration {num_iterations}: Time taken = {end_iteration - start_iteration:.4f} seconds")
            sys.stdout.flush()

        print(f"\nbatch_size : {batch_size}")
        print(f"\nTotal iterations in {CONFIG.inference_time} seconds: {num_iterations}")


if __name__ == "__main__":
    if CONFIG.main_mode == 'train':
        train_main()
    elif CONFIG.main_mode == 'inference':
        inference_time_main()
    else:
        raise ValueError(f"Invalid main mode: {CONFIG.main_mode}. Choose 'train' or 'inference'.")