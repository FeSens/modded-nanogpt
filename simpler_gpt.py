import os
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import glob
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter

# Muon optimizer simplified out, using AdamW

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

def causal_score_mod(score, batch_idx, head_idx, q_idx, k_idx):
    return torch.where(q_idx >= k_idx, score, float("-inf"))

class CausalSelfAttention(nn.Module):
    def __init__(self, config, max_seq_len: int = 1024):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        # Merged QKV weights with improved initialization
        hdim = self.n_head * self.head_dim  # Should be equal to n_embd
        std = 0.5 * (self.n_embd ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, self.n_embd).uniform_(-bound, bound))
        self.c_proj = nn.Linear(hdim, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()  # Zero init
        self.rotary = Rotary(self.head_dim)  # Assuming original Rotary without max_seq_len if not needed
        self.attn_scale = 0.12  # Fixed attention scale

    def forward(self, x):
        B, T, C = x.size()
        # Compute QKV using merged weights
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x))
        qkv = qkv.view(B, T, 3 * self.n_head, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)
        # QK normalization (assuming norm is F.rms_norm)
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        # Apply rotary embeddings (assuming rotary API from second code)
        cos, sin = self.rotary(q)  # Compute cos, sin based on q's shape
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Transpose for heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = flex_attention(q, k, v, score_mod=causal_score_mod, scale=self.attn_scale, kernel_options={
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_M1": 32,
            "BLOCK_N1": 64,
            "BLOCK_M2": 64,
            "BLOCK_N2": 32,
        })
       
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class DyT(nn.Module):
    def __init__(self, config, init_alpha):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(config.n_embd))
        self.beta = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.dyt1 = DyT(config, init_alpha=0.8)
        self.dyt2 = DyT(config, init_alpha=0.2)

    def forward(self, x):
        x = x + self.attn(self.dyt1(x))
        x = x + self.mlp(self.dyt2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    return tokens

class DataLoader:
    def __init__(self, filename_pattern, B, T):
        self.B = B
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

@dataclass
@dataclass
class Hyperparameters:
    input_bin: str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin: str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size: int = 8
    sequence_length: int = 1024
    learning_rate: float = 0.0036
    val_loss_every_n_tokens: int = 48*1024*8*125
    val_tokens: int = 10485760

    def __post_init__(self):
        self.num_iterations = (48*1024*8*1750) // (self.batch_size * self.sequence_length)
        self.warmup_iters = 0
        self.warmdown_iters = int(self.num_iterations * 0.45)

args = Hyperparameters()

device = 'cuda'
# torch.cuda.set_device(device)
B, T = args.batch_size, args.sequence_length
val_steps = args.val_tokens // (B * T)
train_loader = DataLoader(args.input_bin, B, T)
val_loader = DataLoader(args.input_val_bin, B, T)
x, y = train_loader.next_batch()

model = GPT(GPTConfig())
model = model.cuda()
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), fused=True)

def get_lr(it):
    if it < args.warmup_iters:
        return (it + 1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

writer = SummaryWriter(log_dir='runs/gpt_training')
tokens_per_step = B * T
total_tokens = 0
training_time_ms = 0.0
torch.cuda.synchronize()
t0 = time.perf_counter()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if last_step or (args.val_loss_every_n_tokens > 0 and total_tokens % args.val_loss_every_n_tokens == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with ctx:
                _, loss = model(x_val, y_val, return_logits=False)
            val_loss += loss.item()
        val_loss /= val_steps
        print(f'step:{step} val_loss:{val_loss:.4f} total_tokens:{total_tokens} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms')
        writer.add_scalar('Loss/val', val_loss, step)
        writer.add_scalar('Tokens/total', total_tokens, step)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    if last_step:
        break
    model.train()
    with ctx:
        _, loss = model(x, y, return_logits=False)
    x, y = train_loader.next_batch()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    total_tokens += tokens_per_step
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print(f"step:{step+1} train_loss:{loss.item():.4f} total_tokens:{total_tokens} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms")
    writer.add_scalar('Loss/train', loss.item(), step)
    writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], step)
writer.close()