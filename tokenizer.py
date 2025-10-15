import torch
from transformers import GPT2Tokenizer, BertTokenizer
from typing import List, Dict, Any

class TextTokenizer:
    def __init__(self, tokenizer_name: str = "gpt2", vocab_size: int = 32000):
        if "bert" in tokenizer_name.lower():
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use provided vocab_size instead of actual tokenizer vocab size
        self.vocab_size = vocab_size
        
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to token IDs"""
        tokens = self.tokenizer.encode(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tokens = tokens.squeeze(0)
        
        # Clamp tokens to vocab_size to avoid CUDA errors
        tokens = torch.clamp(tokens, 0, self.vocab_size - 1)
        
        return tokens
    
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
        tokens = tokens["input_ids"]
        
        # Clamp tokens to vocab_size to avoid CUDA errors
        tokens = torch.clamp(tokens, 0, self.vocab_size - 1)
        
        return tokens
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
