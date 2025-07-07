"""
Configuration parser for Axon IA.

This module provides utilities for loading, parsing, and validating
configuration files for the framework.
"""

import os
from pathlib import Path
import yaml
from typing import Any, Dict, List, Optional, Union


class ConfigParser:
    """
    Parser for YAML configuration files.
    
    Handles loading, validation, and access to configuration parameters
    with support for nested keys using dot notation.
    """
    
    def __init__(self, config_path: Union[str, Path], validate: bool = True):
        """
        Initialize configuration parser.
        
        Args:
            config_path: Path to the YAML configuration file
            validate: Whether to validate the configuration
        """
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config(self.config_path)
        
        # Validate if requested
        if validate:
            self._validate_config()
    
    def _load_config(self, path: Path) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            Parsed configuration dictionary
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self) -> None:
        """
        Validate the configuration.
        
        Checks for required sections and parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for required top-level sections
        required_sections = ["data", "model", "training"]
        missing_sections = [section for section in required_sections if section not in self.config]
        
        if missing_sections:
            raise ValueError(f"Missing required configuration sections: {', '.join(missing_sections)}")
        
        # Check data section
        data_config = self.config["data"]
        if "root_dir" not in data_config:
            raise ValueError("Missing 'root_dir' in data configuration")
        
        # Check model section
        model_config = self.config["model"]
        if "architecture" not in model_config:
            raise ValueError("Missing 'architecture' in model configuration")
        
        # Check training section
        training_config = self.config["training"]
        if "epochs" not in training_config:
            raise ValueError("Missing 'epochs' in training configuration")
        if "output_dir" not in training_config:
            raise ValueError("Missing 'output_dir' in training configuration")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., "model.params.in_channels")
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default
        """
        if not key:
            return self.config
        
        # Split key into parts
        parts = key.split(".")
        
        # Navigate through config
        curr = self.config
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return default
        
        return curr
    
    def override(self, key: str, value: Any) -> None:
        """
        Override a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation
            value: New value
        """
        # Split key into parts
        parts = key.split(".")
        
        # Navigate through config
        curr = self.config
        for i, part in enumerate(parts[:-1]):
            if part not in curr:
                curr[part] = {}
            curr = curr[part]
        
        # Set the value
        curr[parts[-1]] = value
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            path: Path to save the configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        return dict(self.config)