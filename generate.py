#!/usr/bin/python3

from transformer_utils import BigramLanguageModel
import torch

gpt_model = BigramLanguageModel()
gpt_model.load_state_dict(torch.load("microgpt.pth"))
gpt_model.eval()


context = torch.ones((1, 1), dtype=torch.long)
gpt_model.generate(context, max_tokens=10000, verbose=1)