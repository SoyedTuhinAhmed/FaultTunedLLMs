from config import Config
from model import *
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset
# import torch.nn.functional as F

# 1. Load pretrained weights from Hugging Face Hub
def load_pretrained_weights(model):
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    state_dict = {}
    
    # Load embeddings and tie weights properly
    state_dict['tok_emb.weight'] = hf_model.transformer.wte.weight.data
    # Remove explicit head.weight loading - let weight tying handle it
    state_dict['pos_emb.weight'] = hf_model.transformer.wpe.weight.data
    
    # Embeddings and output head
    state_dict['tok_emb.weight'] = hf_model.transformer.wte.weight.data
    state_dict['pos_emb.weight'] = hf_model.transformer.wpe.weight.data
    state_dict['head.weight'] = hf_model.lm_head.weight.data
    
    # Final layer norm
    state_dict['ln_f.weight'] = hf_model.transformer.ln_f.weight.data
    state_dict['ln_f.bias'] = hf_model.transformer.ln_f.bias.data

    # Transformer blocks
    for blk_idx in range(hf_model.config.n_layer):
        hf_block = hf_model.transformer.h[blk_idx]
        prefix = f'blocks.{blk_idx}.'
        
        # QKV projections
        state_dict[f'{prefix}attn.W_qkv.weight'] = hf_block.attn.c_attn.weight.data.t()
        state_dict[f'{prefix}attn.W_qkv.bias'] = hf_block.attn.c_attn.bias.data
        
        # Output projection
        state_dict[f'{prefix}attn.W_o.weight'] = hf_block.attn.c_proj.weight.data.t()
        state_dict[f'{prefix}attn.W_o.bias'] = hf_block.attn.c_proj.bias.data
        
        # MLP layers
        state_dict[f'{prefix}mlp.linear1.weight'] = hf_block.mlp.c_fc.weight.data.t()
        state_dict[f'{prefix}mlp.linear1.bias'] = hf_block.mlp.c_fc.bias.data
        state_dict[f'{prefix}mlp.linear2.weight'] = hf_block.mlp.c_proj.weight.data.t()
        state_dict[f'{prefix}mlp.linear2.bias'] = hf_block.mlp.c_proj.bias.data
        
        # Layer norms
        state_dict[f'{prefix}ln_1.weight'] = hf_block.ln_1.weight.data
        state_dict[f'{prefix}ln_1.bias'] = hf_block.ln_1.bias.data
        state_dict[f'{prefix}ln_2.weight'] = hf_block.ln_2.weight.data
        state_dict[f'{prefix}ln_2.bias'] = hf_block.ln_2.bias.data

    model.load_state_dict(state_dict)
    return model

# 2. Create model with proper config (GPT-2 small)
# Initialize model
config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2(config).to(device)

# Load weights and evaluate
model = load_pretrained_weights(model)
model.eval()
# After loading weights
print(model.blocks[0].attn.W_qkv.weight.shape)  # Should be [2304, 768]
print(model.blocks[0].attn.W_o.weight.shape)    # Should be [768, 768] 
print(model.blocks[0].mlp.linear1.weight.shape) # Should be [3072, 768]
print(torch.allclose(model.tok_emb.weight, model.head.weight))  # Should be True

# 4. Load tokenizer and dataset
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
# dataset = load_dataset("wikitext", "wikitext-2-v1", split="test")

# # 2. Convert test lines to one big text string, then tokenize
# test_texts = dataset["text"]  # All lines in test set
# full_text = "\n".join(test_texts)
# tokens = tokenizer(full_text, truncation=False)["input_ids"]

# # 3. Chunk tokens into blocks of 1024
# block_size = 1024
# blocks = []
# for i in range(0, len(tokens), block_size):
#     block = tokens[i : i + block_size]
#     if len(block) > 1:  # We'll skip any chunk <2 tokens
#         blocks.append(block)

# @torch.no_grad()
# def evaluate():
#     losses = []
#     tokenizer.pad_token_id = tokenizer.eos_token_id
    
#     for block in blocks:
#         # Convert list of IDs -> tensor of shape [1, seq_len]
#         input_ids = torch.tensor(block, device=device).unsqueeze(0)
        
#         # Forward pass: input_ids[:, :-1] predicts the next token
#         logits = model(input_ids[:, :-1])
        
#         # Shift labels by 1 (predict input_ids[:, 1:])
#         labels = input_ids[:, 1:].contiguous()

#         # Compute cross entropy
#         loss = F.cross_entropy(
#             logits.view(-1, logits.size(-1)), 
#             labels.view(-1),
#             ignore_index=tokenizer.pad_token_id,
#             reduction="mean"
#         )
#         losses.append(loss.item())
    
#     # Calculate perplexity from the average loss
#     mean_loss = torch.tensor(losses).mean()
#     perplexity = torch.exp(mean_loss)
#     return perplexity.item()

# # 4. Run evaluation
# model.eval()  # Make sure dropout is turned off
# ppl = evaluate()
# print(f"Perplexity: {ppl:.2f}")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Use the raw version for proper evaluation
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

# 2. Preprocess with proper truncation
def process_examples(examples):
    return tokenizer(
        examples["text"],
        max_length=1024,
        truncation=True,
        padding=False,
        return_tensors=None
    )

dataset = dataset.map(process_examples, batched=False)

# 3. Filter empty sequences and create chunks
block_size = 1024
valid_sequences = [
    ex["input_ids"][i:i+block_size] 
    for ex in dataset 
        for i in range(0, len(ex["input_ids"]), block_size)
            if len(ex["input_ids"][i:i+block_size]) > 1
]

# 4. Revised evaluation function
# @torch.no_grad()
# def evaluate():
#     losses = []
    
#     for seq in valid_sequences:
#         input_ids = torch.tensor(seq, device=device).unsqueeze(0)
        
#         with torch.inference_mode():
#             logits = model(input_ids[:, :-1])
            
#         loss = F.cross_entropy(
#             logits.view(-1, logits.size(-1)),
#             input_ids[:, 1:].contiguous().view(-1),
#             ignore_index=tokenizer.pad_token_id
#         )
#         losses.append(loss.item())

#     return torch.exp(torch.tensor(losses).mean()).item()

@torch.no_grad()
def evaluate():
    losses = []
    for seq in valid_sequences:
        if len(seq) < 2: continue
        
        input_ids = torch.tensor(seq, device=device).unsqueeze(0)
        logits = model(input_ids[:, :-1])
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        losses.append(loss.item())

    return torch.exp(torch.tensor(losses).mean()).item()

# Run evaluation
model.eval()
ppl = evaluate()
print(f"Perplexity: {ppl:.2f}")  # Should now be ~20-25 without warnings