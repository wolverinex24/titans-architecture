"""
Automatically generated __init__.py for utils

This module exports the following:
- Classes: CheckpointManager, MemoryMetrics, MetricsTracker, ModelConfig, TitansConfig, TrainingConfig
- Functions: compute_accuracy, compute_average, compute_memory_usage, compute_perplexity, from_dict, get_cosine_schedule_with_warmup, get_optimizer_groups, load_best, load_latest, load_yaml, lr_lambda, reset, save, save_yaml, setup_logger, update
"""

from typing import Any, Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from .config import ModelConfig, TitansConfig, TrainingConfig, from_dict, load_yaml, save_yaml
from .optimization import get_cosine_schedule_with_warmup, get_optimizer_groups, lr_lambda
from .metrics import MemoryMetrics, MetricsTracker, compute_accuracy, compute_average, compute_memory_usage, compute_perplexity, reset, update
from .checkpoint import CheckpointManager, load_best, load_latest, save
from .logging import setup_logger

__all__ = [
    'CheckpointManager',
    'MemoryMetrics',
    'MetricsTracker',
    'ModelConfig',
    'TitansConfig',
    'TrainingConfig',
    'compute_accuracy',
    'compute_average',
    'compute_memory_usage',
    'compute_perplexity',
    'from_dict',
    'get_cosine_schedule_with_warmup',
    'get_optimizer_groups',
    'load_best',
    'load_latest',
    'load_yaml',
    'lr_lambda',
    'reset',
    'save',
    'save_yaml',
    'setup_logger',
    'update'
]
