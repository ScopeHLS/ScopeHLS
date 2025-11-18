import torch
import torch.nn as nn
from transformers import AutoModel
from copy import deepcopy

from config.config import CONFIG

class PragmaEncoder(nn.Module):
    """
    Pragma Encoder: Initialized from the first N layers of CodeT5+, trained independently
    Input: token ids + attention mask
    Output: hidden states (B, len, dim)
    """
    
    def __init__(self, codet5p_path=CONFIG.codet5p_path, num_layers=CONFIG.pragma_layers):
        super().__init__()
        self.num_layers = num_layers
        full_model = AutoModel.from_pretrained(codet5p_path)
        self.encoder = self._create_independent_encoder(full_model, num_layers)
        self.embeddings = deepcopy(full_model.shared)
        self.config = full_model.config
        del full_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    def _create_independent_encoder(self, full_model, num_layers):
        independent_encoder = deepcopy(full_model.encoder)
        independent_encoder.block = independent_encoder.block[:num_layers]
        independent_encoder.config.num_layers = num_layers
        return independent_encoder
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward propagation
        
        Args:
            input_ids: torch.Tensor, shape (B, len) - token ids of the pragma
            attention_mask: torch.Tensor, shape (B, len) - attention mask
            
        Returns:
            torch.Tensor: hidden states, shape (B, len, dim)
        """
        inputs_embeds = self.embeddings(input_ids)
        
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        return encoder_outputs.last_hidden_state

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)   
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.dropout = nn.Dropout(CONFIG.dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key, mask, alpha=CONFIG.default_alpha):
        B, Q_len, D = query.size()
        K_len = key.size(1)

        q = self.q_proj(query).view(B, Q_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, Q, d)
        k = self.k_proj(key).view(B, K_len, self.n_heads, self.head_dim).transpose(1, 2)    # (B, h, K, d)
        v = self.v_proj(key).view(B, K_len, self.n_heads, self.head_dim).transpose(1, 2)    # (B, h, K, d)


        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, h, Q, K)

        if mask != None:
            scores_mask = scores * mask # Apply floating-point attention mask (B, Q, K)
            scores = alpha * scores_mask + (1 - alpha) * scores

        scores = scores.clamp(-CONFIG.scores_clamp, CONFIG.scores_clamp)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B, h, Q, d)
        context = context.transpose(1, 2).contiguous().view(B, Q_len, D)
        out = self.out_proj(context)

        return self.norm(query + self.dropout(out))

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=CONFIG.eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class T5DenseActDense(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.wi = nn.Linear(hidden_size, hidden_size*4, bias=False)
        self.wo = nn.Linear(hidden_size*4, hidden_size, bias=False)
        self.dropout = nn.Dropout(p=CONFIG.dropout)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states
    
class T5LayerFF(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.DenseReluDense = T5DenseActDense(hidden_size)
        self.layer_norm = T5LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=CONFIG.dropout)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5LayerSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.layer_norm = T5LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, key_padding_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_output, _ = self.self_attn(hidden_states, hidden_states, hidden_states, key_padding_mask=key_padding_mask)
        return residual + self.dropout(attn_output)


class T5LayerCrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.cross_attn = CrossAttentionBlock(hidden_size, num_heads)

    def forward(self, A_input, B_input, scope_mask, alpha):
        output = self.cross_attn(A_input, B_input, scope_mask, alpha=alpha)
        return output

class ScopeAwareT5Block(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=CONFIG.dropout):
        super().__init__()
        self.code_self_attn = T5LayerSelfAttention(hidden_size, num_heads, dropout)
        self.pragma_self_attn = T5LayerSelfAttention(hidden_size, num_heads, dropout)

        self.cross_code_to_pragma = T5LayerCrossAttention(hidden_size, num_heads)
        self.cross_pragma_to_code = T5LayerCrossAttention(hidden_size, num_heads)

        self.code_ff = T5LayerFF(hidden_size)
        self.pragma_ff = T5LayerFF(hidden_size)

    def forward(self, code_input, pragma_input, code_mask, pragma_mask, scope_mask, alpha):
        code_input = self.code_self_attn(code_input, code_mask)
        pragma_input = self.pragma_self_attn(pragma_input, pragma_mask)

        if scope_mask == None:            
            code_input = self.cross_code_to_pragma(code_input, pragma_input, scope_mask=None, alpha=alpha)
            pragma_input = self.cross_pragma_to_code(pragma_input, code_input, scope_mask=None, alpha=alpha)
        else:
            code_input = self.cross_code_to_pragma(code_input, pragma_input, scope_mask=scope_mask.transpose(-1, -2), alpha=alpha)
            pragma_input = self.cross_pragma_to_code(pragma_input, code_input, scope_mask=scope_mask, alpha=alpha)

        code_input = self.code_ff(code_input)
        pragma_input = self.pragma_ff(pragma_input)

        return code_input, pragma_input
    
class GlobalAttentionPooling(nn.Module):
    """Global attention pooling layer that computes a weighted average of hidden states"""
    def __init__(self, hidden_size, attention_size=None):
        super().__init__()
        if attention_size is None:
            attention_size = hidden_size
        
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        self.attention_net = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1, bias=False)
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)
        Returns:
            pooled_output: (batch_size, hidden_size)
        """
        
        attention_scores = self.attention_net(hidden_states)
        attention_scores = attention_scores.squeeze(-1)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask, -1e9)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        pooled_output = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        
        return pooled_output