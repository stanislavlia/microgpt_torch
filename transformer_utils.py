import numpy as np
import torch
import torch.nn as nn
from torch.functional import F
import os

with open("input.txt", "r") as file:
    text_corpus =   file.read()

unique_chars = sorted(set(text_corpus))


char_to_token = {unique_chars[i]:i for i in range(len(unique_chars))}

token_to_char =  {i:unique_chars[i] for i in range(len(unique_chars))}

#=========
#constants
N_EMBED=128
BATCH_SIZE=64
N_HEADS=8
VOCAB_SIZE=len(unique_chars)
BLOCK_SIZE=36 #Context length
N_BLOCKS=6
MAX_TOKENS=100
LR=0.001
EPOCHS=1000

#=======

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        self.head_size = head_size
        
        self.key_layer = nn.Linear(N_EMBED, head_size, bias=False)
        self.query_layer = nn.Linear(N_EMBED, head_size, bias=False)
        self.value_layer = nn.Linear(N_EMBED, head_size, bias=False)
        
    def forward(self, x):
        
        B, T, C = x.shape
        
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)
        
        #compute attention scores for each token wrt to other tokens
        attention_weights = (query @ key.transpose(-2, -1)) * C ** -0.5
        
        #mask out future
        tril = torch.tril(torch.ones(T, T))
        attention_weights = attention_weights.masked_fill(tril == 0, float("-inf"))
        #apply softmax to transform to prob distribution
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        #gather values of interesting tokens and aggregate them for each token
        
        out = attention_weights @ value
        
        return out
        

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.projection = nn.Linear(N_EMBED, N_EMBED)
        
        
    def forward(self, x):
        
        head_outs = [head(x) for head in self.heads]
        
        #concatenate outputs of heads
        head_outs_concated = torch.cat(head_outs, dim=-1)
        
        out = self.projection(head_outs_concated)
        
        
        return out
    
    
class TransformerBlock(nn.Module): 
    def __init__(self, n_emb, n_heads):
        super().__init__()

        
        self.multihead_attention = MultiHeadAttention(n_heads, n_emb // n_heads)
        
        
        self.feedforward_nn = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
        )
        
        self.layer_norm1  = nn.LayerNorm(n_emb)
        self.layer_norm2 = nn.LayerNorm(n_emb)
        
    def forward(self, x): #output shape is same as input
        
        x = self.multihead_attention(x) + x #add residual connections
        x = self.feedforward_nn(x) + x 
        
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        self.position_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)
        
        #init transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(N_EMBED, N_HEADS) for _ in range(N_BLOCKS)])
        self.layer_norm = nn.LayerNorm(N_EMBED)
        
        self.final_linear = nn.Linear(N_EMBED, VOCAB_SIZE) #logits
        
        
    def forward(self, idx, targets=None):
        #idx - tensor of indices of tokens
        
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) #(B, T, N_EMB)
        pos_emb = self.position_embedding_table(torch.arange(T)) #(T, N_EMB)
        
        #sum token and position embedding
        x = pos_emb + tok_emb
        
        x = self.blocks(x) #(B, T, N_EMB)
        x = self.layer_norm(x) #(B, T, N_EMB)
        logits = self.final_linear(x) #(B, T, VOCAB_SIZE) - produces logits
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            
            logits = logits.view(B*T, C)
            
            targets = targets.view(B*T,)
            
            loss = F.cross_entropy(logits, targets)
        
        
        
        return logits, loss
        

    def generate(self, idx, max_tokens=100, verbose=0):
        
        # idx is (B, T) array of indices in the current context
        
        for _ in range(max_tokens):
            
            #crop last context
            idx_cond = idx[:, -BLOCK_SIZE:]
            
            #get predicitons
            logits, loss = self(idx_cond)
            
            #focus only on last time-step
            logits = logits[:, -1, :] #(B, C)
            
            #apply softmax
            tok_probs = F.softmax(logits, dim=-1)
            
            #sample from dist
            idx_next = torch.multinomial(tok_probs, num_samples=1)
            
            #append new token
            idx = torch.cat((idx, idx_next), dim=-1)
            if (verbose):
                print(token_to_char[int(idx_next)], end="")
            
        return idx
        