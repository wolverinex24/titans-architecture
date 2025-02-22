# titans/scripts/utils/profile_model.py
import torch
from core.models import TitansMAC
from utils.metrics import MetricsTracker
from utils.logging import setup_logger
import time
from typing import Dict, Any
from torch.profiler import profile, record_function, ProfilerActivity

def profile_model_performance(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Profile model performance metrics."""
    results = {}
    
    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = model(sample_input)
    results['peak_memory'] = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Throughput
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(sample_input)
    avg_time = (time.time() - start_time) / num_runs
    results['tokens_per_second'] = sample_input.size(1) / avg_time
    
    # Detailed profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True
    ) as prof:
        with record_function("model_inference"):
            _ = model(sample_input)
            
    results['profile'] = prof
    
    return results