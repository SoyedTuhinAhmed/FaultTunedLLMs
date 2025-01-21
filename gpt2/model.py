import torch, torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = self.d_model // self.num_heads
        
        # Combined QKV projection (matches HF implementation)
        self.W_qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project to QKV in one matrix multiplication
        qkv = self.W_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.view(1, 1, seq_len, seq_len)
            
        # Apply mask before softmax
        scores = scores.masked_fill(mask == 0, -1e9)  # Use large negative value instead of -inf
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Combine heads and project back
        attn_output = (attn_weights @ v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.resid_dropout(self.W_o(attn_output))
        
        return attn_output

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.mlp = FeedForward(config)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_pos_embeddings, config.d_model)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Token and positional embeddings
        token_embeddings = self.tok_emb(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeddings = self.pos_emb(position_ids)
        x = self.drop(token_embeddings + position_embeddings)

        # Mask to prevent attending to future tokens
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).view(1, 1, seq_len, seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm and prediction head
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
