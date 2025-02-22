# titans/utils/memory_monitor.py

import torch
import numpy as np
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import deque

@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a particular time."""
    step: int
    memory_state: torch.Tensor
    surprise_value: float
    momentum_value: torch.Tensor
    forget_gate_value: torch.Tensor
    memory_usage: float  # Percentage of active memory cells
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for serialization."""
        return {
            'step': self.step,
            'memory_state': self.memory_state.cpu().numpy().tolist(),
            'surprise_value': float(self.surprise_value),
            'momentum_value': self.momentum_value.cpu().numpy().tolist(),
            'forget_gate_value': self.forget_gate_value.cpu().numpy().tolist(),
            'memory_usage': float(self.memory_usage)
        }

class MemoryMonitor:
    """Monitor and analyze memory behavior during training and inference."""
    
    def __init__(
        self,
        log_dir: str = 'memory_logs',
        max_snapshots: int = 1000,
        memory_dim: int = 512
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('memory_monitor')
        
        # Initialize storage
        self.snapshots = deque(maxlen=max_snapshots)
        self.training_metrics = []
        self.inference_metrics = []
        
        # Running statistics
        self.memory_usage_stats = {
            'min': float('inf'),
            'max': float('-inf'),
            'avg': 0.0,
            'count': 0
        }
        
        # Memory dimension for analysis
        self.memory_dim = memory_dim
        
    def take_snapshot(
        self,
        step: int,
        memory_state: torch.Tensor,
        surprise_value: float,
        momentum_value: torch.Tensor,
        forget_gate_value: torch.Tensor
    ) -> None:
        """Take a snapshot of current memory state."""
        # Calculate memory usage (percentage of active cells)
        memory_usage = torch.count_nonzero(memory_state) / memory_state.numel()
        
        snapshot = MemorySnapshot(
            step=step,
            memory_state=memory_state.detach(),
            surprise_value=float(surprise_value),
            momentum_value=momentum_value.detach(),
            forget_gate_value=forget_gate_value.detach(),
            memory_usage=float(memory_usage)
        )
        
        # Update running statistics
        self._update_stats(memory_usage)
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
    def _update_stats(self, memory_usage: float) -> None:
        """Update running statistics."""
        stats = self.memory_usage_stats
        stats['min'] = min(stats['min'], memory_usage)
        stats['max'] = max(stats['max'], memory_usage)
        stats['avg'] = (stats['avg'] * stats['count'] + memory_usage) / (stats['count'] + 1)
        stats['count'] += 1
        
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.snapshots:
            return {}
            
        # Convert snapshots to numpy arrays for analysis
        memory_states = np.array([s.memory_state.cpu().numpy() for s in self.snapshots])
        surprise_values = np.array([s.surprise_value for s in self.snapshots])
        memory_usage = np.array([s.memory_usage for s in self.snapshots])
        
        analysis = {
            'memory_usage': {
                'mean': float(np.mean(memory_usage)),
                'std': float(np.std(memory_usage)),
                'min': float(np.min(memory_usage)),
                'max': float(np.max(memory_usage))
            },
            'surprise_patterns': {
                'mean': float(np.mean(surprise_values)),
                'std': float(np.std(surprise_values)),
                'peaks': self._detect_surprise_peaks(surprise_values)
            },
            'memory_stability': self._analyze_memory_stability(memory_states),
            'forget_gate_activity': self._analyze_forget_gate_activity()
        }
        
        return analysis
        
    def _detect_surprise_peaks(self, surprise_values: np.ndarray) -> List[int]:
        """Detect significant peaks in surprise values."""
        # Simple peak detection using local maxima
        peaks = []
        if len(surprise_values) > 2:
            for i in range(1, len(surprise_values) - 1):
                if (surprise_values[i] > surprise_values[i-1] and 
                    surprise_values[i] > surprise_values[i+1] and
                    surprise_values[i] > np.mean(surprise_values) + np.std(surprise_values)):
                    peaks.append(i)
        return peaks
        
    def _analyze_memory_stability(self, memory_states: np.ndarray) -> Dict[str, float]:
        """Analyze stability of memory states over time."""
        if len(memory_states) < 2:
            return {'state_changes': 0.0}
            
        # Calculate state changes between consecutive steps
        changes = np.abs(np.diff(memory_states, axis=0))
        stability_metric = float(np.mean(changes))
        
        return {
            'state_changes': stability_metric,
            'stable_cells_ratio': float(np.mean(changes < 0.1))  # Cells that change less than 0.1
        }
        
    def _analyze_forget_gate_activity(self) -> Dict[str, float]:
        """Analyze forget gate behavior."""
        forget_gates = np.array([s.forget_gate_value.cpu().numpy() for s in self.snapshots])
        
        if len(forget_gates) == 0:
            return {}
            
        return {
            'mean_forget_rate': float(np.mean(forget_gates)),
            'forget_frequency': float(np.mean(forget_gates > 0.5))  # How often cells are forgotten
        }
        
    def plot_memory_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot memory-related metrics."""
        if not self.snapshots:
            self.logger.warning("No snapshots available for plotting")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Memory Usage
        memory_usage = [s.memory_usage for s in self.snapshots]
        axes[0,0].plot(memory_usage)
        axes[0,0].set_title('Memory Usage Over Time')
        axes[0,0].set_ylabel('Usage Ratio')
        
        # Surprise Values
        surprise_values = [s.surprise_value for s in self.snapshots]
        axes[0,1].plot(surprise_values)
        axes[0,1].set_title('Surprise Values')
        
        # Forget Gate Activity
        forget_gates = [s.forget_gate_value.mean().item() for s in self.snapshots]
        axes[1,0].plot(forget_gates)
        axes[1,0].set_title('Average Forget Gate Activity')
        
        # Memory State Changes
        if len(self.snapshots) > 1:
            states = torch.stack([s.memory_state for s in self.snapshots])
            changes = torch.abs(states[1:] - states[:-1]).mean(dim=1)
            axes[1,1].plot(changes.cpu().numpy())
            axes[1,1].set_title('Memory State Changes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def save_analysis(self, path: str) -> None:
        """Save memory analysis to file."""
        analysis = self.analyze_memory_patterns()
        
        # Save analysis
        with open(path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        # Save plots
        plot_path = Path(path).parent / 'memory_plots.png'
        self.plot_memory_metrics(str(plot_path))
        
    def log_memory_status(self, step: int) -> None:
        """Log current memory status."""
        if not self.snapshots:
            return
            
        latest = self.snapshots[-1]
        self.logger.info(
            f"Step {step} - Memory Usage: {latest.memory_usage:.3f}, "
            f"Surprise: {latest.surprise_value:.3f}, "
            f"Avg Forget Rate: {latest.forget_gate_value.mean().item():.3f}"
        )