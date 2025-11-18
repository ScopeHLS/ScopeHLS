import torch
import torch.nn as nn
from tqdm import tqdm

def compute_loss(outputs, targets, device):
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3).to(device))
    mse_loss = nn.MSELoss()
    norm_constants = {
        'LUT': 1,
        'FF': 1,
        'DSP': 1,
        'BRAM': 1,
    }
    
    losses = {}
    for key in outputs:
        if key not in targets:
            continue
        out = outputs[key].squeeze()
        tgt = targets[key].squeeze()
        if key in norm_constants:
            norm = norm_constants[key]
            out = out / norm
            tgt = tgt / norm

        if key == 'valid':
            losses[key] = bce_loss(out, tgt)
        else:
            losses[key] = mse_loss(out, tgt)

    total_loss = sum(losses.values())
    losses['total'] = total_loss
    return losses

def print_losses(epoch, losses, prefix="Train"):
    """
    Print individual sub-losses and total loss.

    Args:
        epoch (int): Current epoch number
        losses (dict): Dictionary returned by compute_loss, containing individual loss items and 'total'
        prefix (str): Output prefix, default is "Train", can be "Val" or others
    """
    total_loss = losses.get('total', None)
    valid_loss = losses.get('valid', 0.0)
    LUT_loss = losses.get('LUT', 0.0)
    FF_loss = losses.get('FF', 0.0)
    DSP_loss = losses.get('DSP', 0.0)
    BRAM_loss = losses.get('BRAM', 0.0)
    perf_loss = losses.get('perf', 0.0)

    if isinstance(total_loss, torch.Tensor):
        total_loss = total_loss.item()
    if isinstance(valid_loss, torch.Tensor):
        valid_loss = valid_loss.item()
    if isinstance(LUT_loss, torch.Tensor):
        LUT_loss = LUT_loss.item()
    if isinstance(FF_loss, torch.Tensor):
        FF_loss = FF_loss.item()
    if isinstance(DSP_loss, torch.Tensor):
        DSP_loss = DSP_loss.item()
    if isinstance(BRAM_loss, torch.Tensor):
        BRAM_loss = BRAM_loss.item()
    if isinstance(perf_loss, torch.Tensor):
        perf_loss = perf_loss.item()

    print(f"Epoch {epoch}: {prefix} Loss = {total_loss:.4f} | "
          f"valid: {valid_loss:.4f}, LUT: {LUT_loss:.4f}, FF: {FF_loss:.4f}, "
          f"DSP: {DSP_loss:.4f}, BRAM: {BRAM_loss:.4f}, perf: {perf_loss:.4f}")

def train_one_epoch(model, dataloader, optimizer, device, mode):
    total_losses = {
        'total': 0.0,
        'valid': 0.0,
        'LUT': 0.0,
        'FF': 0.0,
        'DSP': 0.0,
        'BRAM': 0.0,
        'perf': 0.0,
    }

    count = 0

    pbar = tqdm(dataloader, desc=f"Training ({mode})", leave=False)
    for batch in pbar:
        optimizer.zero_grad()

        outputs = model(
            code_input_ids=batch['code_tokens']['input_ids'].to(device),
            pragma_input_ids=batch['param_tokens']['input_ids'].to(device),
            code_mask=batch['code_tokens']['attention_mask'].to(device),
            pragma_mask=batch['param_tokens']['attention_mask'].to(device),
            scope_mask=batch['scope_mask'].to(device),
            mode=mode
        )

        targets = {
            'valid': batch['valid'].float().unsqueeze(-1).to(device),
            'LUT': batch['res_util']['util-LUT'].float().unsqueeze(-1).to(device),
            'FF': batch['res_util']['util-FF'].float().unsqueeze(-1).to(device),
            'DSP': batch['res_util']['util-DSP'].float().unsqueeze(-1).to(device),
            'BRAM': batch['res_util']['util-BRAM'].float().unsqueeze(-1).to(device),
            'perf': batch['perf'].float().unsqueeze(-1).to(device),
        }

        losses = compute_loss(outputs, targets, device)
        total_loss = losses['total']

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            loss_value = losses.get(k, 0)
            total_losses[k] += loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value

        count += 1
        pbar.set_postfix({k: f"{(total_losses[k] / count):.4f}" for k in total_losses if k != "total"})

    for k in total_losses:
        total_losses[k] /= count

    return total_losses

def validate(model, dataloader, device, mode):
    total_losses = {
        'total': 0.0,
        'valid': 0.0,
        'LUT': 0.0,
        'FF': 0.0,
        'DSP': 0.0,
        'BRAM': 0.0,
        'perf': 0.0,
    }
    count = 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Validate ({mode})", leave=False)
        for batch in pbar:
            outputs = model(
                code_input_ids=batch['code_tokens']['input_ids'],
                pragma_input_ids=batch['param_tokens']['input_ids'],
                code_mask=batch['code_tokens']['attention_mask'] ,
                pragma_mask=batch['param_tokens']['attention_mask'],
                scope_mask=batch['scope_mask'],
                mode=mode
            )

            targets = {
                'valid': batch['valid'].float().unsqueeze(-1).to(device),
                'LUT': batch['res_util']['util-LUT'].float().unsqueeze(-1).to(device),
                'FF': batch['res_util']['util-FF'].float().unsqueeze(-1).to(device),
                'DSP': batch['res_util']['util-DSP'].float().unsqueeze(-1).to(device),
                'BRAM': batch['res_util']['util-BRAM'].float().unsqueeze(-1).to(device),
                'perf': batch['perf'].float().unsqueeze(-1).to(device),
            }


            losses = compute_loss(outputs, targets, device)
            for k in total_losses:
                total_losses[k] += losses.get(k, 0).item() if isinstance(losses.get(k, 0), torch.Tensor) else losses.get(k, 0)
            count += 1
            pbar.set_postfix({k: f"{(total_losses[k] / count):.4f}" for k in total_losses if k != "total"})
        for k in total_losses:
            total_losses[k] /= count
    return total_losses
