import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import os
import random
import numpy as np
from llama import LlamaModel, LlamaConfig
from tokenizer import TextTokenizer
from data_utils import create_dataloaders
from trainer import Trainer

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    
    # Create directories
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = TextTokenizer(
        tokenizer_name=cfg.data.tokenizer_name,
        vocab_size=cfg.data.vocab_size
    )
    
    # Create model
    print("Creating model...")
    model_config = LlamaConfig(
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        vocab_size=cfg.data.vocab_size,
        hidden_size=cfg.model.hidden_size,
        seq_len=cfg.data.max_length
    )
    
    model = LlamaModel(model_config)
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize trainer
    trainer = Trainer(model, cfg, cfg.device)
    
    # Start training
    trainer.train(train_loader, val_loader)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
