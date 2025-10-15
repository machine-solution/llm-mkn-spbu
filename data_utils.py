import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Dict, Any, List
from tokenizer import TextTokenizer

class WikiDataset(Dataset):
    def __init__(self, dataset_name: str, tokenizer: TextTokenizer, max_length: int = 512, split: str = "train"):
        self.dataset = load_dataset(dataset_name, split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["text"]
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, self.max_length)
        
        # Clamp tokens to vocab_size to avoid CUDA errors
        # Crutch
        vocab_size = self.tokenizer.get_vocab_size()
        tokens = torch.clamp(tokens, 0, vocab_size - 1)
        
        # For language modeling, input and target are the same (shifted by 1)
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        
        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "text": text
        }

def create_dataloaders(config: Dict[str, Any], tokenizer: TextTokenizer) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""
    
    # Load full dataset
    full_dataset = WikiDataset(
        dataset_name=config["data"]["dataset_name"],
        tokenizer=tokenizer,
        max_length=config["data"]["max_length"],
        split=config["data"]["train_split"]
    )
    
    # Split into train/val
    val_size = int(len(full_dataset) * config["data"]["val_split_size"])
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"]
    )
    
    return train_loader, val_loader
