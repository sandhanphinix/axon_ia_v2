"""
Comprehensive logging system for the Axon IA package.

Provides detailed, configurable logging across all components.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter


class AxonLogger:
    """
    Advanced logger for the Axon IA package with multi-backend support.
    
    Features:
    - Console logging with colored output
    - File logging with rotation
    - TensorBoard integration
    - Weights & Biases integration (optional)
    - Performance monitoring
    - Memory usage tracking
    """
    
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    
    def __init__(
        self,
        name: str,
        log_dir: Union[str, Path] = "logs",
        level: int = logging.INFO,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        log_gpu_stats: bool = True,
        log_memory_usage: bool = True,
        log_to_file: bool = True,
    ):
        """
        Initialize the logger with multiple backends.
        
        Args:
            name: Logger name
            log_dir: Directory to store logs
            level: Logging level
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B team/entity
            wandb_config: W&B configuration
            log_gpu_stats: Whether to log GPU statistics
            log_memory_usage: Whether to log memory usage
            log_to_file: Whether to log to file
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            tb_log_dir = self.log_dir / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_log_dir))
            self.info(f"TensorBoard logging enabled at {tb_log_dir}")
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                if wandb_project is None:
                    wandb_project = name
                
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config,
                    dir=str(self.log_dir / "wandb"),
                    name=f"{name}_{timestamp}"
                )
                self.info(f"Weights & Biases logging enabled for project {wandb_project}")
            except ImportError:
                self.warning("wandb not installed. W&B logging disabled.")
                self.use_wandb = False
        
        # GPU/Memory monitoring
        self.log_gpu_stats = log_gpu_stats and torch.cuda.is_available()
        self.log_memory_usage = log_memory_usage
        
        self.info(f"Logger initialized: {name}")
        if self.log_gpu_stats:
            device_count = torch.cuda.device_count()
            self.info(f"Found {device_count} GPU(s)")
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                self.info(f"GPU {i}: {gpu_name}, Memory: {total_memory:.2f} GB")
        
        # Start time for performance tracking
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to TensorBoard and/or W&B."""
        for name, value in metrics.items():
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(name, value, step)
            
            if self.use_wandb:
                try:
                    import wandb
                    if step is not None:
                        wandb.log({name: value, 'step': step})
                    else:
                        wandb.log({name: value})
                except (ImportError, Exception):
                    pass
        
        # Log system metrics
        if self.log_gpu_stats and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                
                if self.tb_writer is not None:
                    self.tb_writer.add_scalar(f'system/gpu{i}/memory_allocated_gb', memory_allocated, step)
                    self.tb_writer.add_scalar(f'system/gpu{i}/memory_reserved_gb', memory_reserved, step)
                
                if self.use_wandb:
                    try:
                        import wandb
                        wandb.log({
                            f'system/gpu{i}/memory_allocated_gb': memory_allocated,
                            f'system/gpu{i}/memory_reserved_gb': memory_reserved
                        }, step=step)
                    except (ImportError, Exception):
                        pass
    
    def log_model(self, model: torch.nn.Module, input_size: tuple = None):
        """Log model architecture to TensorBoard and/or W&B."""
        if self.tb_writer is not None and input_size is not None:
            # Create a sample input tensor
            sample_input = torch.zeros(input_size, device=next(model.parameters()).device)
            self.tb_writer.add_graph(model, sample_input)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.watch(model, log="all")
            except (ImportError, Exception):
                pass
    
    def log_image(self, name: str, image, step: int = None):
        """Log image to TensorBoard and/or W&B."""
        if self.tb_writer is not None:
            self.tb_writer.add_image(name, image, step)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({name: wandb.Image(image)}, step=step)
            except (ImportError, Exception):
                pass
    
    def log_elapsed_time(self, name: str, step: int = None, reset: bool = True):
        """Log elapsed time since last call or initialization."""
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        
        self.info(f"{name} completed in {elapsed:.2f} seconds")
        
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(f'time/{name}', elapsed, step)
        
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f'time/{name}': elapsed}, step=step)
            except (ImportError, Exception):
                pass
        
        if reset:
            self.last_log_time = current_time
        
        return elapsed
    
    def close(self):
        """Clean up resources."""
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except (ImportError, Exception):
                pass
        
        # Final timing log
        total_runtime = time.time() - self.start_time
        hours = int(total_runtime // 3600)
        minutes = int((total_runtime % 3600) // 60)
        seconds = total_runtime % 60
        
        self.info(f"Total runtime: {hours}h {minutes}m {seconds:.2f}s")


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[41m\033[97m',  # White on red background
        'RESET': '\033[0m'  # Reset color
    }
    
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


# Singleton logger instance
_logger_instance = None


def get_logger(name: str = "axon_ia", log_dir: Union[str, Path] = "logs", level=logging.INFO) -> AxonLogger:
    """Get or create the logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AxonLogger(name, log_dir, level=level)
    return _logger_instance