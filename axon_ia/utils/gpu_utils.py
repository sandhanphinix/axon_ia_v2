"""
Utilities for GPU management and optimization.

This module provides functions for monitoring GPU usage,
optimizing memory allocation, and managing devices.
"""

import os
import time
from typing import List, Dict, Optional, Tuple, Union

import torch
import numpy as np


def get_available_devices() -> List[torch.device]:
    """
    Get a list of available GPU devices.
    
    Returns:
        List of available torch.device objects
    """
    devices = []
    
    # Add CPU
    devices.append(torch.device("cpu"))
    
    # Add GPUs if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f"cuda:{i}"))
    
    return devices


def get_device_info() -> Dict[str, Dict]:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {"cpu": {"name": "CPU", "memory_total": 0, "memory_available": 0}}
    
    # Add GPU information if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / (1024**3)  # Convert to GB
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # Convert to GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)  # Convert to GB
            memory_free = memory_total - memory_allocated
            
            info[f"cuda:{i}"] = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "memory_total": memory_total,
                "memory_allocated": memory_allocated,
                "memory_reserved": memory_reserved,
                "memory_free": memory_free,
                "multi_processor_count": props.multi_processor_count
            }
    
    return info


def select_best_device() -> torch.device:
    """
    Select the best available device based on memory and compute capability.
    
    Returns:
        Best torch.device object
    """
    # If no CUDA, return CPU
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    # Get device info
    info = get_device_info()
    
    # Filter CUDA devices
    cuda_devices = {k: v for k, v in info.items() if k.startswith("cuda")}
    if not cuda_devices:
        return torch.device("cpu")
    
    # Find device with most free memory
    best_device = max(cuda_devices.items(), key=lambda x: x[1]["memory_free"])
    
    return torch.device(best_device[0])


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get memory statistics for a device.
    
    Args:
        device: Device to get stats for, or None for current device
        
    Returns:
        Dictionary with memory statistics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    stats = {}
    
    if device.type == "cuda":
        # Get device index
        device_idx = device.index if device.index is not None else 0
        
        # Get memory stats
        stats["allocated"] = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
        stats["reserved"] = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
        stats["max_allocated"] = torch.cuda.max_memory_allocated(device_idx) / (1024**3)  # GB
        stats["max_reserved"] = torch.cuda.max_memory_reserved(device_idx) / (1024**3)  # GB
        
        # Get device properties
        props = torch.cuda.get_device_properties(device_idx)
        stats["total"] = props.total_memory / (1024**3)  # GB
        stats["free"] = stats["total"] - stats["allocated"]
    
    return stats


def empty_cache():
    """Empty CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_memory_fraction(fraction: float = 0.9):
    """
    Set maximum memory fraction to use.
    
    Args:
        fraction: Fraction of total memory to use (0.0-1.0)
    """
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, i)
            except AttributeError:
                # Fallback for older PyTorch versions
                device = torch.device(f"cuda:{i}")
                try:
                    torch.cuda.memory.set_per_process_memory_fraction(fraction, device)
                except AttributeError:
                    # If neither method is available, log the issue
                    from axon_ia.utils.logger import get_logger
                    logger = get_logger()
                    logger.warning(f"Could not set memory fraction on GPU {i} - feature not available in this PyTorch version")


def optimize_gpu_memory():
    """Apply various optimizations to reduce GPU memory usage."""
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        
        # Set memory fraction
        set_memory_fraction(0.9)
        
        # Try enabling TF32 if available (on Ampere GPUs)
        if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmark for optimized kernels
        torch.backends.cudnn.benchmark = True


def get_optimal_batch_size(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
    start_batch_size: int = 32,
    target_memory_usage: float = 0.8,
) -> int:
    """
    Find the optimal batch size for a model based on memory usage.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (without batch dimension)
        device: Device to use (default: current model device)
        start_batch_size: Initial batch size to try
        target_memory_usage: Target memory usage fraction
        
    Returns:
        Optimal batch size
    """
    if device is None:
        device = next(model.parameters()).device
    
    if device.type != "cuda":
        return start_batch_size  # For CPU, just return the starting value
    
    # Move model to device if needed
    model = model.to(device)
    model.eval()
    
    # Get device memory
    device_info = get_device_info()[str(device)]
    total_memory = device_info["memory_total"]
    target_memory = total_memory * target_memory_usage
    
    # Start with given batch size
    batch_size = start_batch_size
    
    # Try reducing batch size until it fits
    while batch_size > 1:
        try:
            # Create dummy input
            dummy_input = torch.zeros((batch_size,) + input_shape, device=device)
            
            # Clear cache and measure before
            empty_cache()
            memory_before = get_memory_stats(device)["allocated"]
            
            # Forward pass
            with torch.no_grad():
                _ = model(dummy_input)
            
            # Measure after
            memory_after = get_memory_stats(device)["allocated"]
            memory_used = memory_after - memory_before
            
            # Check if memory usage is within target
            if memory_after < target_memory:
                # Memory usage is OK, try increasing batch size
                next_batch_size = batch_size * 2
                
                # Create dummy input with larger batch size
                dummy_input = torch.zeros((next_batch_size,) + input_shape, device=device)
                
                # Clear cache and try forward pass
                empty_cache()
                with torch.no_grad():
                    _ = model(dummy_input)
                
                # If no error, update batch size
                batch_size = next_batch_size
            else:
                # Memory usage is too high, reduce batch size
                batch_size = batch_size // 2
                
        except RuntimeError as e:
            # Out of memory, reduce batch size
            batch_size = batch_size // 2
            empty_cache()
    
    # Ensure minimum batch size of 1
    return max(1, batch_size)


def benchmark_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict[str, float]:
    """
    Benchmark a model's inference speed.
    
    Args:
        model: PyTorch model
        input_shape: Shape of a single input (without batch dimension)
        batch_size: Batch size to use
        device: Device to use (default: current model device)
        n_warmup: Number of warm-up runs
        n_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Move model to device if needed
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.zeros((batch_size,) + input_shape, device=device)
    
    # Warm-up runs
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark runs
    latencies = []
    start = time.time()
    for _ in range(n_runs):
        run_start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        latencies.append(time.time() - run_start)
    
    total_time = time.time() - start
    
    # Calculate statistics
    latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds
    avg_latency = np.mean(latencies_ms)
    std_latency = np.std(latencies_ms)
    min_latency = np.min(latencies_ms)
    max_latency = np.max(latencies_ms)
    throughput = n_runs * batch_size / total_time
    
    return {
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_samples_per_sec": throughput,
        "device": str(device)
    }