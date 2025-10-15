import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from typing import Dict, Any, Tuple
import math

class Trainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            betas=(config["training"]["beta1"], config["training"]["beta2"])
        )
        
        # Setup scheduler
        total_steps = config["training"]["num_epochs"] * config["training"]["batch_size"]
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config["training"]["min_lr"]
        )
        
        # Setup TensorBoard
        self.writer = SummaryWriter(log_dir=config["log_dir"])
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Calculate loss (cross entropy)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config["training"]["max_grad_norm"]
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def validate(self, val_loader) -> float:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                
                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_ids.view(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train_epoch(self, train_loader, val_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        # Check if we have max_steps limit
        max_steps = self.config["training"].get("max_steps")
        if max_steps is not None and self.global_step >= max_steps:
            return 0.0, 0.0  # Already reached max steps
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch in progress_bar:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            self.global_step += 1
            
            # Log metrics
            if self.global_step % self.config["training"]["log_every"] == 0:
                self.writer.add_scalar("train/loss", loss, self.global_step)
                self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                
                progress_bar.set_postfix({
                    "loss": f"{loss:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    "step": f"{self.global_step}"
                })
            
            # Save checkpoint
            if self.global_step % self.config["training"]["save_every"] == 0:
                self.save_checkpoint()
            
            # Check max_steps limit
            if max_steps is not None and self.global_step >= max_steps:
                print(f"\nReached max_steps limit: {max_steps}")
                break
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_val_loss = self.validate(val_loader)
        
        # Log epoch metrics
        self.writer.add_scalar("epoch/train_loss", avg_train_loss, self.epoch)
        self.writer.add_scalar("epoch/val_loss", avg_val_loss, self.epoch)
        
        return avg_train_loss, avg_val_loss
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        max_steps = self.config["training"].get("max_steps")
        
        if max_steps is not None:
            print(f"Starting training for max {max_steps} steps")
        else:
            print(f"Starting training for {self.config['training']['num_epochs']} epochs")
        
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config["training"]["num_epochs"]):
            self.epoch = epoch
            train_loss, val_loss = self.train_epoch(train_loader, val_loader)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Log epoch metrics
            self.writer.add_scalar("epoch/train_loss", train_loss, epoch)
            self.writer.add_scalar("epoch/val_loss", val_loss, epoch)
            
            # Check if we reached max_steps
            if max_steps is not None and self.global_step >= max_steps:
                print(f"Training stopped: reached max_steps ({max_steps})")
                break
        
        # Save final model
        self.save_checkpoint(is_final=True)
        self.writer.close()
    
    def save_checkpoint(self, is_final: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config
        }
        
        if is_final:
            path = os.path.join(self.config["checkpoint_dir"], "final_model.pt")
        else:
            path = os.path.join(self.config["checkpoint_dir"], f"checkpoint_step_{self.global_step}.pt")
        
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
