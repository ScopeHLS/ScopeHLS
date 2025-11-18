import torch
import torch.nn as nn
from transformers import AutoModel

from config.config import CONFIG
from model.model_base import ScopeAwareT5Block, GlobalAttentionPooling, PragmaEncoder


class pragma_cross_code(nn.Module):
    def __init__(self):
        super().__init__()
        # print("make model--load pretrained codet5p-770m encoder")

        self.codet5 = AutoModel.from_pretrained(CONFIG.codet5p_path)
        self.encoder = self.codet5.encoder
        self.use_encoder = True

        self.pragma_encoder = PragmaEncoder()

        self.t5_dim = CONFIG.t5_dim
        self.hidden_dim = CONFIG.hidden_dim
        self.n_heads = CONFIG.self_n_heads
        self.codet5_our = nn.Linear(self.t5_dim, self.hidden_dim)
        self.alpha = CONFIG.default_alpha
        self.attention_pooling_size = CONFIG.attention_pooling_size

        self.decoder_layers = CONFIG.decoder_layers

        self.scope_blocks = nn.ModuleList([
            ScopeAwareT5Block(hidden_size=self.hidden_dim, num_heads=self.n_heads, dropout=CONFIG.dropout)
            for _ in range(self.decoder_layers)
            ])
        
        self.code_attention_pooling = GlobalAttentionPooling(self.hidden_dim, self.attention_pooling_size)
        self.pragma_attention_pooling = GlobalAttentionPooling(self.hidden_dim, self.attention_pooling_size)

        # MLP heads: 6 branches
        self.freeze_mlp = [False] * 6  # control switches

        self.valid_head = self._make_head(1)
        
        self.LUT_head = self._make_head(1)
        self.FF_head = self._make_head(1)
        self.DSP_head = self._make_head(1)
        self.BRAM_head = self._make_head(1)

        self.perf_head = self._make_head(1)


    def _freeze_head(self, head):
        for param in head.parameters():
            param.requires_grad = False

    def _make_head(self, out_dim):
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim)
        )
    
    def update_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, code_input_ids=None, pragma_input_ids=None, code_input=None, pragma_input=None,
                code_mask=None, pragma_mask=None, scope_mask=None, mode=None):
        
        code_input_ids, pragma_input_ids, code_input, pragma_input, \
        code_mask, pragma_mask, scope_mask, code_pad_mask, pragma_pad_mask, \
        code_attention_mask, pragma_attention_mask, device = self.prepare(
            code_input_ids, pragma_input_ids, code_input, pragma_input,
            code_mask, pragma_mask, scope_mask, mode
        )

        if mode is None or mode == 'inference':
            outputs = self.cache_4_inference(
                pragma_input_ids=pragma_input_ids,
                code_input_cache=code_input,
                code_mask=code_mask,
                pragma_mask=pragma_mask,
                scope_mask=scope_mask
            )
            return outputs
        else:
            outputs = self.train_forward(
                code_input_ids=code_input_ids,
                pragma_input_ids=pragma_input_ids,
                code_input=code_input,
                pragma_input=pragma_input,
                code_mask=code_mask,
                pragma_mask=pragma_mask,
                scope_mask=scope_mask,
                mode=mode,
                code_pad_mask=code_pad_mask,
                pragma_pad_mask=pragma_pad_mask,
                code_attention_mask=code_attention_mask,
                pragma_attention_mask=pragma_attention_mask
            )
            return outputs
        
    def get_cached_inputs(self, code_input_ids=None, code_mask=None):
        
        device = next(self.parameters()).device
        def safe_to_device(tensor, device):
            return tensor.to(device) if tensor is not None else None
        code_input_ids = safe_to_device(code_input_ids, device)
        code_mask = safe_to_device(code_mask, device)

        code_input = self.encoder(
            input_ids=code_input_ids,
            attention_mask=code_mask
        ).last_hidden_state
        return code_input

    
    def cache_4_inference(self, pragma_input_ids=None, code_input_cache=None, 
                          code_mask=None, pragma_mask=None, scope_mask=None):

        code_pad_mask=(code_mask == 0)
        pragma_pad_mask=(pragma_mask == 0)
        code_attention_mask = ~code_pad_mask
        pragma_attention_mask = ~pragma_pad_mask

        pragma_input = self.pragma_encoder(
            input_ids=pragma_input_ids,
            attention_mask=pragma_mask
        )

        code_input = self.codet5_our(code_input_cache)
        pragma_input = self.codet5_our(pragma_input)

        for i, block in enumerate(self.scope_blocks):
            if i in CONFIG.no_scope_blocks:
                code_input, pragma_input = block(
                    code_input, pragma_input,
                    code_pad_mask, pragma_pad_mask,
                    None,
                    self.alpha
                )
            else:
                code_input, pragma_input = block(
                    code_input, pragma_input,
                    code_pad_mask, pragma_pad_mask,
                    scope_mask,
                    self.alpha
                )

        code_feat = self.code_attention_pooling(code_input, code_attention_mask)
        pragma_feat = self.pragma_attention_pooling(pragma_input, pragma_attention_mask)
        final_feat = pragma_feat + code_feat

        outputs = {}
        outputs['valid'] = self.valid_head(final_feat)
        outputs['LUT'] = self.LUT_head(final_feat)
        outputs['FF'] = self.FF_head(final_feat)
        outputs['DSP'] = self.DSP_head(final_feat)
        outputs['BRAM'] = self.BRAM_head(final_feat)
        outputs['perf'] = self.perf_head(final_feat)

        return outputs
    
    def train_forward(self, code_input_ids=None, pragma_input_ids=None, code_input=None, pragma_input=None,
                        code_mask=None, pragma_mask=None, scope_mask=None, mode=None,
                        code_pad_mask=None, pragma_pad_mask=None, code_attention_mask=None, pragma_attention_mask=None):
        
        if self.use_encoder and code_input is None and pragma_input is None:
            code_input = self.encoder(
                input_ids=code_input_ids,
                attention_mask=code_mask
            ).last_hidden_state
            pragma_input = self.pragma_encoder(
                input_ids=pragma_input_ids,
                attention_mask=pragma_mask
            )

        code_input = self.codet5_our(code_input)
        pragma_input = self.codet5_our(pragma_input)

        for i, block in enumerate(self.scope_blocks):
            if i in CONFIG.no_scope_blocks:
                code_input, pragma_input = block(
                    code_input, pragma_input,
                    code_pad_mask, pragma_pad_mask,
                    None,
                    self.alpha
                )
            else:
                code_input, pragma_input = block(
                    code_input, pragma_input,
                    code_pad_mask, pragma_pad_mask,
                    scope_mask,
                    self.alpha
                )

        code_feat = self.code_attention_pooling(code_input, code_attention_mask)
        pragma_feat = self.pragma_attention_pooling(pragma_input, pragma_attention_mask)

        final_feat = pragma_feat + code_feat
        if mode is None:
            self.freeze_mlp = [False] * 6
        elif mode == 'stage1':
            self.freeze_mlp[0] = False
            for i in range(5):
                self.freeze_mlp[i+1] = True
        elif mode == 'stage2':
            self.freeze_mlp[0] = True
            for i in range(5):
                self.freeze_mlp[i+1] = False
        elif mode == 'stage3':
            self.freeze_mlp = [False] * 6
        else:
            raise ValueError("mode must be one of ['stage1', 'stage2', 'stage3', None]")

        if self.freeze_mlp[0]:
            self._freeze_head(self.valid_head)
        if self.freeze_mlp[1]:
            self._freeze_head(self.LUT_head)
        if self.freeze_mlp[2]:
            self._freeze_head(self.FF_head)
        if self.freeze_mlp[3]:
            self._freeze_head(self.DSP_head)
        if self.freeze_mlp[4]:
            self._freeze_head(self.BRAM_head)
        if self.freeze_mlp[5]:
            self._freeze_head(self.perf_head)

        outputs = {}

        out_valid = self.valid_head(final_feat)
        out_LUT = self.LUT_head(final_feat)
        out_FF = self.FF_head(final_feat)
        out_DSP = self.DSP_head(final_feat)
        out_BRAM = self.BRAM_head(final_feat)
        out_perf = self.perf_head(final_feat)

        if self.freeze_mlp[0]:
            outputs['valid'] = out_valid.detach()
        else:
            outputs['valid'] = out_valid

        if self.freeze_mlp[1]:
            outputs['LUT'] = torch.relu(out_LUT.detach())
        else:
            outputs['LUT'] = torch.relu(out_LUT)

        if self.freeze_mlp[2]:
            outputs['FF'] = torch.relu(out_FF.detach())
        else:
            outputs['FF'] = torch.relu(out_FF)

        if self.freeze_mlp[3]:
            outputs['DSP'] = torch.relu(out_DSP.detach())
        else:
            outputs['DSP'] = torch.relu(out_DSP)

        if self.freeze_mlp[4]:
            outputs['BRAM'] = torch.relu(out_BRAM.detach())
        else:
            outputs['BRAM'] = torch.relu(out_BRAM)

        if self.freeze_mlp[5]:
            outputs['perf'] = torch.tanh(out_perf.detach()) * CONFIG.perf_scale
        else:
            outputs['perf'] = torch.tanh(out_perf) * CONFIG.perf_scale

        return outputs
        

    def prepare(self, code_input_ids=None, pragma_input_ids=None, code_input=None, pragma_input=None,
                code_mask=None, pragma_mask=None, scope_mask=None, mode=None):
        device = next(self.parameters()).device

        def safe_to_device(tensor, device):
            return tensor.to(device) if tensor is not None else None
        
        def safe_squeeze(tensor, dim):
            return tensor.squeeze(dim) if tensor is not None else None

        code_input_ids = safe_to_device(code_input_ids, device)
        code_mask = safe_to_device(code_mask, device)
        pragma_input_ids = safe_to_device(pragma_input_ids, device)
        pragma_mask = safe_to_device(pragma_mask, device)
        scope_mask = safe_to_device(scope_mask, device)
        code_input = safe_to_device(code_input, device)

        # Ensure inputs are 3D tensors
        code_input_ids = safe_squeeze(code_input_ids, 1)  # [B, 1, code_len] → [B, code_len]
        pragma_input_ids = safe_squeeze(pragma_input_ids, 1)  # [B, 1, paragma_len] → [B, paragma_len]
        code_mask = safe_squeeze(code_mask, 1)  # [B, 1, code_len] → [B, code_len]
        pragma_mask = safe_squeeze(pragma_mask, 1)  # [B, 1, paragma_len] → [B, paragma_len]
        code_input = safe_squeeze(code_input, 1)  # [B, 1, code_len, t5_dim] → [B, code_len, t5_dim]
        
        code_pad_mask=(code_mask == 0)
        pragma_pad_mask=(pragma_mask == 0)

        code_attention_mask = ~code_pad_mask
        pragma_attention_mask = ~pragma_pad_mask
        scope_mask = scope_mask.unsqueeze(1)
        scope_mask = scope_mask.expand(-1, self.n_heads, -1, -1)

        return code_input_ids, pragma_input_ids, code_input, pragma_input, \
                code_mask, pragma_mask, scope_mask, code_pad_mask, pragma_pad_mask, \
                code_attention_mask, pragma_attention_mask, device
    
    def freeze_by_mode(self, mode):
        """
        Freeze model components based on training stage.
        stage1: freeze code_encoder (self.encoder)
        stage2: unfreeze code_encoder, freeze valid_head
        stage3: unfreeze all
        """
        if mode == "stage1":
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif mode == "stage2":
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            raise ValueError("mode must be one of ['stage1', 'stage2']")
