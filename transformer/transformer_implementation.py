import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fix: Use Matplotlib Agg backend to avoid GTK issues
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid segmentation faults

# Define a Simple Transformer Model
class SimpleTransformer(nn.Module):
    def _init_(self, input_dim=50, hidden_dim=128, num_heads=4, num_layers=2, output_dim=10, max_seq_len=100):
        super(SimpleTransformer, self)._init_()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        self.attention_weights = []  # Store attention weights as PyTorch tensors
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        self.attention_weights.clear()  # Reset previous attention weights
        for layer in self.encoder.layers:
            attn = layer.self_attn(x, x, x, need_weights=True)[1]  # Extract attention
            self.attention_weights.append(attn.detach())  # Fix: Store as a tensor, not NumPy array
        x = self.encoder(x)
        x = self.fc(x[:, 0, :])  # Use the first token's output
        return x

# Create model and dummy input
device = torch.device("cpu")  # Change to "cuda" if using GPU
model = SimpleTransformer().to(device)
input_data = torch.randint(0, 50, (1, 10), dtype=torch.int64).to(device)

# Performance Benchmarking
def measure_toks_per_second(model, input_tensor, num_iters=100):
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(input_tensor)
    elapsed_time = time.time() - start_time
    toks_per_sec = (num_iters * input_tensor.shape[1]) / elapsed_time
    return toks_per_sec

toks_per_sec = measure_toks_per_second(model, input_data)
print(f"⚡ Tokens Per Second (CPU): {toks_per_sec:.2f} toks/s")

# Visualize Attention Weights
if model.attention_weights:
    attn_map = model.attention_weights[-1][0].cpu().numpy()  # Fix: Convert to NumPy only at visualization step
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_map, cmap="viridis", annot=False)
    plt.title("Transformer Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.savefig("attention_heatmap.png")  # Fix: Save instead of showing
    print("✅ Attention heatmap saved as attention_heatmap.png")
