import math
import torch
from dataclasses import dataclass

from torch import nn
from torch.nn import Linear

@dataclass
class LlamaConfig():
    n_layers: int
    n_heads: int
    vocab_size: int
    hidden_size: int
    seq_len: int


class SelfAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        self.WQ = Linear(config.hidden_size, config.hidden_size)
        self.WK = Linear(config.hidden_size, config.hidden_size)
        self.WV = Linear(config.hidden_size, config.hidden_size)
        self.WO = Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        q = self.WQ(x)
        k = self.WK(x)
        v = self.WV(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = nn.functional.softmax(
            torch.where(
                torch.triu(torch.ones(seq_len, seq_len)).to(x.device).to(torch.bool),
                -torch.inf,
                (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.head_dim)),
            ),
            dim=-1,
        )
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        out = self.WO(out)
        return out

class RoPE(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.base = 10000.0
        self.head_dim = config.hidden_size // config.n_heads
        self.t = self.base ** (- 2 * (torch.arange(0, self.head_dim // 2, dtype=torch.float32) - 1) / (self.head_dim))
        self.range = torch.arange(0, self.seq_len, dtype=torch.float32)

        self.temp = self.range.view(-1, 1) * self.t.view(1, -1)

        self.cos = torch.cos(self.temp)
        self.sin = torch.sin(self.temp)

    def split(self, x):
        first_half = x[:, :, :self.head_dim // 2]
        second_half = x[:, :, self.head_dim // 2:]
        return first_half, second_half
    
    def rotate(self, x):
        batch_size, seq_len = x.shape[:2]
        range_pos = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
        temp = range_pos.view(-1, 1) * self.t.view(1, -1).to(x.device)
        cos = torch.cos(temp)
        sin = torch.sin(temp)
        
        first_half, second_half = self.split(x)
        # cos and sin have shape (seq_len, head_dim//2), we need to broadcast to (batch_size, seq_len, head_dim//2)
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
        
        return torch.cat(
            [
                first_half * cos + second_half * sin, 
                first_half * sin - second_half * cos
            ], 
            dim=-1
        )

    def forward(self, q, k):
        return self.rotate(q), self.rotate(k)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.gate_proj = Linear(config.hidden_size, config.hidden_size * 4)
        self.up_proj = Linear(config.hidden_size, config.hidden_size * 4)
        self.down_proj = Linear(config.hidden_size * 4, config.hidden_size)

    def forward(self, x):
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.attention = SelfAttention(config)
        self.rope = RoPE(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        normed_x = self.attention_norm(x)
        q = self.attention.WQ(normed_x)
        k = self.attention.WK(normed_x)
        v = self.attention.WV(normed_x)
        
        q = q.view(batch_size, seq_len, self.attention.n_heads, self.attention.head_dim)
        k = k.view(batch_size, seq_len, self.attention.n_heads, self.attention.head_dim)
        v = v.view(batch_size, seq_len, self.attention.n_heads, self.attention.head_dim)
        
        for i in range(self.attention.n_heads):
            q[:, :, i, :], k[:, :, i, :] = self.rope(q[:, :, i, :], k[:, :, i, :])
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = nn.functional.softmax(
            torch.where(
                torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device).to(torch.bool),
                -torch.inf,
                (q @ k.transpose(-2, -1)) * (1 / math.sqrt(self.attention.head_dim)),
            ),
            dim=-1,
        )
        attn_out = attn @ v
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_out = self.attention.WO(attn_out)
        
        x = x + attn_out
        
        normed_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(normed_x)
        
        x = x + ffn_out
        
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_length - len(input_ids)):
                logits = self.forward(input_ids)[-1:]
                
                logits = logits / temperature
                
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits = torch.full_like(logits, -float('inf'))
                    logits.scatter_(-1, top_k_indices, top_k_logits)
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=0)
        
        return input_ids
