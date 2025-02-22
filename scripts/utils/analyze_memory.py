# titans/scripts/utils/analyze_memory.py
import torch
from core.models import TitansMAC
from core.memory import NeuralMemoryModule
from utils.metrics import MemoryMetrics
from utils.logging import setup_logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

def analyze_memory_states(
    memory_states: List[torch.Tensor],
    save_dir: str
):
    """Analyze and visualize memory states."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute statistics
    activity = torch.stack([
        (torch.abs(state) > 0.1).float().mean(dim=-1)
        for state in memory_states
    ])
    
    entropy = torch.stack([
        -(torch.softmax(state, dim=-1) * 
          torch.log_softmax(state, dim=-1)).sum(dim=-1)
        for state in memory_states
    ])
    
    # Plot memory activity
    plt.figure(figsize=(12, 6))
    sns.heatmap(activity.cpu().numpy(), cmap='viridis')
    plt.title('Memory Activity Over Time')
    plt.xlabel('Memory Position')
    plt.ylabel('Time Step')
    plt.savefig(save_dir / 'memory_activity.png')
    plt.close()
    
    # Plot entropy
    plt.figure(figsize=(12, 6))
    plt.plot(entropy.mean(dim=-1).cpu().numpy())
    plt.title('Memory Entropy Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Entropy')
    plt.savefig(save_dir / 'memory_entropy.png')
    plt.close()