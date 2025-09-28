import torch
from transformers import GPT2Tokenizer
from typing import List, Dict, Any

class TextTokenizer:
    def __init__(self, tokenizer_name: str = "gpt2", vocab_size: int = 32000):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.vocab_size = len(self.tokenizer)
        
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to token IDs"""
        tokens = self.tokenizer.encode(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokens.squeeze(0)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def batch_encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Encode batch of texts"""
        tokens = self.tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return tokens["input_ids"]
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
