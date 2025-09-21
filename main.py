import torch
from llama import LlamaModel, LlamaConfig

def test_llama_model():
    config = LlamaConfig(
        n_layers=2,
        n_heads=4,
        vocab_size=1000,
        hidden_size=128,
        seq_len=32
    )
    
    model = LlamaModel(config)

    input_ids = torch.randint(0, config.vocab_size, (config.seq_len,))
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Forward pass: logits shape = {logits.shape}")
    print(f"Logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
    print(f"Contains NaN: {torch.isnan(logits).any().item()}")
    
    model.train()
    logits = model(input_ids)
    loss = torch.mean(logits)
    loss.backward()
    
    total_params = sum(p.numel() for p in model.parameters())
    params_with_grad = sum(p.numel() for p in model.parameters() if p.grad is not None)
    
    print(f"Backward pass: loss = {loss.item():.6f}")
    print(f"Parameters with gradients: {params_with_grad}/{total_params}")
    
    return model, config

if __name__ == "__main__":
    model, config = test_llama_model()
