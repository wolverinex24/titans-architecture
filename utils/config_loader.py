# titans/utils/config_loader.py
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy

class ConfigLoader:
    def __init__(self, config_dir: str):
        """Configuration loader for Titans.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.model_configs_dir = self.config_dir / "model_configs"
        
        # Load default configuration
        self.default_config_path = self.config_dir / "default_config.yaml"
        if self.default_config_path.exists():
            self.default_config = self._load_yaml(self.default_config_path)
        else:
            raise FileNotFoundError(f"Default config not found at {self.default_config_path}")
            
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
            
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        merged = deepcopy(base)
        
        for key, value in override.items():
            if (
                key in merged and 
                isinstance(merged[key], dict) and 
                isinstance(value, dict)
            ):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = deepcopy(value)
                
        return merged
            
    def load_model_config(
        self,
        model_size: str = "base",
        override_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Load model configuration with optional overrides."""
        # Start with default configuration
        config = deepcopy(self.default_config)
        
        # Load model size specific configuration
        model_config_path = self.model_configs_dir / f"{model_size}.yaml"
        if model_config_path.exists():
            model_config = self._load_yaml(model_config_path)
            config = self._merge_configs(config, model_config)
            
        # Apply override configuration if provided
        if override_config:
            config = self._merge_configs(config, override_config)
            
        return config