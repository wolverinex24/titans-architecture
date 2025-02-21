"""
Automatically generated __init__.py for models

This module exports the following:
- Classes: TitansMAC
- Functions: create_causal_mask, forward, get_model_config
"""

from typing import Any, Dict, Optional, Tuple
from .mac import TitansMAC, forward
from .model_utils import create_causal_mask, get_model_config

__all__ = [
    'TitansMAC',
    'create_causal_mask',
    'forward',
    'get_model_config'
]
